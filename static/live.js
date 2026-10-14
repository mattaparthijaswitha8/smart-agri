/* LIVE dashboard: decision tree, K-Means, regression, explorer - all interactive (d3, offline) */
const CFG = window.CFG;
const $ = s => document.querySelector(s);
const S = { rain: 800, temp: 25, soil: CFG.soils.includes("Loamy") ? "Loamy" : CFG.soils[0] };
const FEATS = { r: "Rainfall (mm)", t: "Temperature (°C)", p: "Soil pH", h: "Humidity (%)", y: "Yield (t/ha)" };
const ZPAL = ["#1565c0", "#c62828", "#2e7d32", "#f9a825", "#6a1b9a", "#00838f", "#ef6c00", "#ad1457"];
let SAMPLE = [];

/* ---------------- global controls ---------------- */
let debounce;
function readControls() {
  S.rain = +$("#rain").value; S.temp = +$("#temp").value; S.soil = $("#soil").value;
  $("#rv").textContent = S.rain; $("#tv").textContent = S.temp;
}
function onControls() { readControls(); clearTimeout(debounce); debounce = setTimeout(refreshAll, 120); }
["rain", "temp", "soil"].forEach(i => $("#" + i).addEventListener("input", onControls));
function refreshAll() { updateRec(); updateBars(); updatePath(); kmMarker(); loadReg(); }

/* ---------------- 0. recommendation panel ---------------- */
async function updateRec() {
  const d = await (await fetch("/api/recommend", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rainfall: S.rain, temperature: S.temp, soil: S.soil }) })).json();
  if (d.error) return;
  $("#recbox").innerHTML = `<div class="best"><small>Best crop for your field</small><div class="name">${d.best.crop}</div>
    <small>Suitability ${d.best.probability}% · Expected yield <b>${d.best.yield_t_ha} t/ha</b></small>
    <p class="muted" style="margin:10px 0 0">K-Means zone: <span class="tag">${d.zone}</span></p></div>
    <div><h3>Top suitable crops</h3>${d.results.map((r, i) => `<div class="row"><span>${i + 1}. ${r.crop}</span><span><b>${r.probability}%</b> · ${r.yield_t_ha} t/ha</span></div>`).join("")}</div>`;
}

/* ---------------- 1. live suitability / yield bars ---------------- */
async function updateBars() {
  const q = new URLSearchParams({ rain: S.rain, temp: S.temp, soil: S.soil });
  const d = await (await fetch("/api/all_crops?" + q)).json();
  const byProb = [...d].sort((a, b) => b.prob - a.prob);
  barChart("#bars-prob", byProb.map(r => ({ key: r.crop, label: r.crop, value: r.prob, color: CFG.colors[r.crop], text: r.prob + "%" })), { max: 100 });
  const byYield = [...d].sort((a, b) => b.index - a.index);
  barChart("#bars-yield", byYield.map(r => ({ key: r.crop, label: r.crop, value: r.index, color: CFG.colors[r.crop], text: r.yield_t_ha + " t/ha" })), { max: 130, right: 90 });
  const best = byProb[0];
  if (!window.regTouched) { const sel = $("#rcrop"); if (sel.value !== best.crop) { sel.value = best.crop; loadReg(); loadTrend(); } }
  $("#bestcrop").innerHTML = `Best crop now: <b>${best.crop}</b> (${best.prob}% · ${best.yield_t_ha} t/ha)`;
}

/* ---------------- 2. live decision tree ---------------- */
let TREE = null, expanded = new Set();
const featName = f => f === "rainfall_mm" ? "Rainfall" : f === "avg_temp_c" ? "Temp" : "Soil";
function featVal(f) { return f === "rainfall_mm" ? S.rain : f === "avg_temp_c" ? S.temp : (S.soil === f.slice(5) ? 1 : 0); }
function nodeText(n) {
  if (!n.f) return `${n.cls} ${(n.p * 100).toFixed(0)}% (n=${n.n})`;
  if (n.f === "rainfall_mm") return `Rainfall ≤ ${n.th.toFixed(0)} mm?`;
  if (n.f === "avg_temp_c") return `Temp ≤ ${n.th.toFixed(1)} °C?`;
  return `Soil is ${n.f.slice(5)}?`;
}
function pathNodes() {
  const out = []; let n = TREE.tree;
  while (n) { out.push(n); if (!n.f) break; n = featVal(n.f) <= n.th ? n.ch[0] : n.ch[1]; }
  return out;
}
async function loadTree() {
  const depth = $("#tdepth").value, leaf = $("#tleaf").value;
  $("#tdv").textContent = depth;
  TREE = await (await fetch(`/api/tree?depth=${depth}&leaf=${leaf}`)).json();
  expanded = new Set();
  (function walk(n, d) { if (n.f && d < 2) { expanded.add(n.id); n.ch.forEach(c => walk(c, d + 1)); } })(TREE.tree, 0);
  $("#tstats").innerHTML = `Test accuracy <b>${TREE.acc_test}%</b> · Train accuracy ${TREE.acc_train}% · Depth ${TREE.depth} · Leaves ${TREE.leaves} · Nodes ${TREE.nodes}`;
  const imp = Object.entries(TREE.importance).filter(e => e[1] > 0.001).sort((a, b) => b[1] - a[1]);
  barChart("#timp", imp.map(e => ({ key: e[0], label: e[0].replace("rainfall_mm", "Rainfall").replace("avg_temp_c", "Temperature").replace("soil_", "Soil: "),
    value: e[1] * 100, color: "#2e7d32", text: (e[1] * 100).toFixed(1) + "%" })), { max: 100, rowH: 24, left: 110, width: 460 });
  updatePath();
}
function updatePath() { if (!TREE) return; pathNodes().forEach(n => { if (n.f) expanded.add(n.id); }); renderTree(); }
function renderTree() {
  const path = pathNodes(), pset = new Set(path.map(n => n.id));
  const leaf = path[path.length - 1];
  $("#tpath").innerHTML = "<b>Decision path for your field:</b> " + path.map(n => {
    if (!n.f) return `<span class="tag" style="background:${CFG.colors[n.cls]};color:#fff">→ ${n.cls} (${(n.p * 100).toFixed(0)}%)</span>`;
    const v = featVal(n.f), yes = v <= n.th;
    return `<span class="tag">${nodeText(n).replace("?", "")} → ${n.f.startsWith("soil_") ? (v ? "yes" : "no") : (yes ? "yes" : "no")}</span>`;
  }).join(" ");
  const root = d3.hierarchy(TREE.tree, n => (n.f && expanded.has(n.id)) ? n.ch : null);
  const dx = 26, dy = 235;
  d3.tree().nodeSize([dx, dy])(root);
  let x0 = Infinity, x1 = -Infinity; root.each(d => { x0 = Math.min(x0, d.x); x1 = Math.max(x1, d.x); });
  const W = (root.height + 1) * dy + 260, H = x1 - x0 + dx * 2;
  const svg = d3.select("#tree").attr("width", W).attr("height", H);
  svg.selectAll("*").remove();
  const g = svg.append("g").attr("transform", `translate(30,${-x0 + dx})`);
  const off = d => 14 + 9 * Math.sqrt(d.data.n / TREE.tree.n) + 7 * (nodeText(d.data).length + (expanded.has(d.data.id) ? 0 : 2));
  const link = d => { const sx = d.source.y + off(d.source), sy = d.source.x, tx = d.target.y - 12, ty = d.target.x, mx = (sx + tx) / 2;
    return `M${sx},${sy}C${mx},${sy} ${mx},${ty} ${tx},${ty}`; };
  g.selectAll("path.l").data(root.links()).join("path").attr("class", "l").attr("fill", "none")
    .attr("d", link).attr("stroke", d => pset.has(d.source.data.id) && pset.has(d.target.data.id) ? "#d32f2f" : "#b0bec5")
    .attr("stroke-width", d => pset.has(d.source.data.id) && pset.has(d.target.data.id) ? 3 : 1.2);
  g.selectAll("text.e").data(root.links()).join("text").attr("class", "e").attr("font-size", 10).attr("fill", "#607d8b")
    .attr("x", d => d.target.y - 52).attr("y", d => d.target.x - 8)
    .text(d => { const left = d.source.data.ch[0] === d.target.data, soil = d.source.data.f.startsWith("soil_"); return soil ? (left ? "No" : "Yes") : (left ? "yes (≤)" : "no (>)"); });
  const tot = TREE.tree.n;
  const node = g.selectAll("g.n").data(root.descendants()).join("g").attr("class", "n")
    .attr("transform", d => `translate(${d.y},${d.x})`).style("cursor", d => d.data.f ? "pointer" : "default")
    .on("click", (e, d) => { if (!d.data.f) return; expanded.has(d.data.id) ? expanded.delete(d.data.id) : expanded.add(d.data.id); renderTree(); });
  node.append("circle").attr("r", d => 5 + 9 * Math.sqrt(d.data.n / tot))
    .attr("fill", d => CFG.colors[d.data.cls]).attr("fill-opacity", d => d.data.f ? 0.35 + 0.65 * d.data.p : 1)
    .attr("stroke", d => pset.has(d.data.id) ? "#000" : "#455a64").attr("stroke-width", d => pset.has(d.data.id) ? 3 : 1)
    .attr("stroke-dasharray", d => d.data.f && !expanded.has(d.data.id) ? "3 2" : null);
  node.append("title").text(d => `${d.data.cls} ${(d.data.p * 100).toFixed(0)}% | samples ${d.data.n}\n` + d.data.top.map(t => `${t[0]}: ${(t[1] * 100).toFixed(0)}%`).join("\n"));
  node.append("text").attr("x", d => 9 + 9 * Math.sqrt(d.data.n / tot)).attr("dy", "0.32em").attr("font-size", 12)
    .attr("font-weight", d => pset.has(d.data.id) ? 700 : 400).attr("fill", "#1b2a1b")
    .attr("stroke", "#fff").attr("stroke-width", 3).attr("paint-order", "stroke")
    .text(d => (d.data.f && !expanded.has(d.data.id) ? "▸ " : "") + nodeText(d.data));
}
$("#tdepth").addEventListener("input", loadTree); $("#tleaf").addEventListener("change", loadTree);
$("#tcollapse").onclick = () => { expanded = new Set([TREE.tree.id]); renderTree(); };
$("#texpand").onclick = () => { (function walk(n, d) { if (n.f && d < 4) { expanded.add(n.id); n.ch.forEach(c => walk(c, d + 1)); } })(TREE.tree, 0); renderTree(); };

/* ---------------- 3. live K-Means (runs in the browser, step by step) ---------------- */
const KM = { k: 4, fx: "r", fy: "t", P: [], cent: [], assign: [], iter: 0, hist: [], trails: [], timer: null, done: false, mu: {}, sd: {} };
const km = { W: 640, H: 440, m: { l: 58, r: 16, t: 14, b: 48 } };
function kmZ(p, f) { return (p[f] - KM.mu[f]) / KM.sd[f]; }
function kmInv(z, f) { return z * KM.sd[f] + KM.mu[f]; }
function kmDist(p, c) { const a = kmZ(p, KM.fx) - c[0], b = kmZ(p, KM.fy) - c[1]; return a * a + b * b; }
function kmAssign() {
  let changed = false, inertia = 0;
  KM.P.forEach((p, i) => {
    let best = 0, bd = Infinity;
    KM.cent.forEach((c, j) => { const d = kmDist(p, c); if (d < bd) { bd = d; best = j; } });
    if (KM.assign[i] !== best) changed = true; KM.assign[i] = best; inertia += bd;
  });
  return { changed, inertia };
}
function kmReset() {
  kmStop();
  KM.k = +$("#kk").value; KM.fx = $("#kfx").value; KM.fy = $("#kfy").value;
  $("#kkv").textContent = KM.k;
  KM.P = SAMPLE.slice(0, 1500);
  [KM.fx, KM.fy].forEach(f => { KM.mu[f] = d3.mean(KM.P, p => p[f]); KM.sd[f] = d3.deviation(KM.P, p => p[f]) || 1; });
  // k-means++ initialisation
  const first = KM.P[Math.floor(Math.random() * KM.P.length)];
  KM.cent = [[kmZ(first, KM.fx), kmZ(first, KM.fy)]];
  while (KM.cent.length < KM.k) {
    const d2 = KM.P.map(p => d3.min(KM.cent, c => kmDist(p, c)));
    let r = Math.random() * d3.sum(d2), idx = 0;
    for (; idx < d2.length - 1; idx++) { r -= d2[idx]; if (r <= 0) break; }
    KM.cent.push([kmZ(KM.P[idx], KM.fx), kmZ(KM.P[idx], KM.fy)]);
  }
  KM.iter = 0; KM.done = false; KM.assign = []; KM.trails = KM.cent.map(c => [[kmInv(c[0], KM.fx), kmInv(c[1], KM.fy)]]);
  KM.hist = [kmAssign().inertia];
  kmDrawAxes(); kmDraw();
}
function kmStep() {
  if (KM.done) { kmStop(); return; }
  const sums = KM.cent.map(() => [0, 0, 0]);
  KM.P.forEach((p, i) => { const s = sums[KM.assign[i]]; s[0] += kmZ(p, KM.fx); s[1] += kmZ(p, KM.fy); s[2]++; });
  KM.cent = KM.cent.map((c, j) => {
    if (!sums[j][2]) { const p = KM.P[Math.floor(Math.random() * KM.P.length)]; return [kmZ(p, KM.fx), kmZ(p, KM.fy)]; }
    return [sums[j][0] / sums[j][2], sums[j][1] / sums[j][2]];
  });
  KM.cent.forEach((c, j) => KM.trails[j].push([kmInv(c[0], KM.fx), kmInv(c[1], KM.fy)]));
  const r = kmAssign(); KM.iter++; KM.hist.push(r.inertia);
  if (!r.changed) { KM.done = true; kmStop(); }
  kmDraw();
}
function kmPlay() { kmStop(); if (KM.done) kmReset(); KM.timer = setInterval(kmStep, +$("#kspeed").value); $("#kplay").textContent = "⏸ Pause"; }
function kmStop() { clearInterval(KM.timer); KM.timer = null; $("#kplay").textContent = "▶ Play"; }
function kmDrawAxes() {
  const svg = d3.select("#kmsvg"); svg.selectAll("*").remove();
  svg.attr("viewBox", `0 0 ${km.W} ${km.H}`);
  km.x = d3.scaleLinear(d3.extent(KM.P, p => p[KM.fx]), [km.m.l, km.W - km.m.r]).nice();
  km.y = d3.scaleLinear(d3.extent(KM.P, p => p[KM.fy]), [km.H - km.m.b, km.m.t]).nice();
  svg.append("g").attr("transform", `translate(0,${km.H - km.m.b})`).call(d3.axisBottom(km.x));
  svg.append("g").attr("transform", `translate(${km.m.l},0)`).call(d3.axisLeft(km.y));
  svg.append("text").attr("x", km.W / 2).attr("y", km.H - 8).attr("text-anchor", "middle").attr("font-size", 12).text(FEATS[KM.fx]);
  svg.append("text").attr("transform", "rotate(-90)").attr("x", -km.H / 2).attr("y", 14).attr("text-anchor", "middle").attr("font-size", 12).text(FEATS[KM.fy]);
  ["pts", "trails", "cents", "mark"].forEach(c => svg.append("g").attr("class", c));
}
function kmDraw() {
  const svg = d3.select("#kmsvg");
  svg.select(".pts").selectAll("circle").data(KM.P).join("circle").attr("r", 3)
    .attr("cx", p => km.x(p[KM.fx])).attr("cy", p => km.y(p[KM.fy]))
    .attr("fill", (p, i) => ZPAL[KM.assign[i] % ZPAL.length]).attr("fill-opacity", .55);
  const line = d3.line().x(d => km.x(d[0])).y(d => km.y(d[1]));
  svg.select(".trails").selectAll("path").data(KM.trails).join("path").attr("fill", "none")
    .attr("stroke", (d, j) => ZPAL[j % ZPAL.length]).attr("stroke-width", 2).attr("stroke-dasharray", "4 3").attr("d", line);
  svg.select(".cents").selectAll("path").data(KM.cent).join("path")
    .attr("d", d3.symbol(d3.symbolCross, 260)()).attr("fill", (d, j) => ZPAL[j % ZPAL.length])
    .attr("stroke", "#000").attr("stroke-width", 2)
    .attr("transform", c => `translate(${km.x(kmInv(c[0], KM.fx))},${km.y(kmInv(c[1], KM.fy))}) rotate(45)`);
  $("#kinfo").innerHTML = `Iteration <b>${KM.iter}</b> · Inertia <b>${KM.hist[KM.hist.length - 1].toFixed(0)}</b> · ` +
    (KM.done ? `<span style="color:#2e7d32"><b>✔ Converged</b> - clusters are stable</span>` : "moving centroids…");
  kmInertia(); kmTable(); kmMarker();
}
function kmInertia() {
  const svg = d3.select("#kminertia"); svg.selectAll("*").remove(); svg.attr("viewBox", "0 0 320 190");
  const m = { l: 48, r: 12, t: 12, b: 32 };
  const x = d3.scaleLinear([0, Math.max(KM.hist.length - 1, 3)], [m.l, 320 - m.r]);
  const y = d3.scaleLinear([0, d3.max(KM.hist)], [190 - m.b, m.t]).nice();
  svg.append("g").attr("transform", `translate(0,${190 - m.b})`).call(d3.axisBottom(x).ticks(5).tickFormat(d3.format("d")));
  svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5));
  svg.append("text").attr("x", 180).attr("y", 186).attr("text-anchor", "middle").attr("font-size", 11).text("iteration");
  svg.append("path").datum(KM.hist).attr("fill", "none").attr("stroke", "#2e7d32").attr("stroke-width", 2.5)
    .attr("d", d3.line().x((d, i) => x(i)).y(d => y(d)));
  svg.selectAll("circle").data(KM.hist).join("circle").attr("r", 3.5).attr("fill", "#2e7d32").attr("cx", (d, i) => x(i)).attr("cy", d => y(d));
}
function kmTable() {
  let h = "<table class='tbl'><tr><th>Cluster</th><th>Points</th><th>Rain</th><th>Temp</th><th>pH</th><th>Top crops</th></tr>";
  for (let j = 0; j < KM.k; j++) {
    const pts = KM.P.filter((p, i) => KM.assign[i] === j);
    if (!pts.length) continue;
    const cnt = d3.rollups(pts, v => v.length, p => p.c).sort((a, b) => b[1] - a[1]).slice(0, 2);
    h += `<tr><td style="color:${ZPAL[j % ZPAL.length]}"><b>● ${j + 1}</b></td><td>${pts.length}</td><td>${d3.mean(pts, p => p.r).toFixed(0)}</td>` +
         `<td>${d3.mean(pts, p => p.t).toFixed(1)}</td><td>${d3.mean(pts, p => p.p).toFixed(1)}</td>` +
         `<td style="text-align:left">${cnt.map(c => `${c[0]} ${(c[1] / pts.length * 100).toFixed(0)}%`).join(", ")}</td></tr>`;
  }
  $("#ktable").innerHTML = h + "</table>";
}
function kmMarker() {
  const g = d3.select("#kmsvg .mark"); if (g.empty()) return; g.selectAll("*").remove();
  const val = f => f === "r" ? S.rain : f === "t" ? S.temp : f === "p" ? CFG.phBySoil[S.soil] : null;
  const vx = val(KM.fx), vy = val(KM.fy);
  if (vx === null || vy === null) { $("#kfield").textContent = "Choose rainfall / temperature / soil pH axes to see where your field lands."; return; }
  const z = [(vx - KM.mu[KM.fx]) / KM.sd[KM.fx], (vy - KM.mu[KM.fy]) / KM.sd[KM.fy]];
  let best = 0, bd = Infinity; KM.cent.forEach((c, j) => { const d = (c[0] - z[0]) ** 2 + (c[1] - z[1]) ** 2; if (d < bd) { bd = d; best = j; } });
  g.append("path").attr("d", d3.symbol(d3.symbolStar, 330)()).attr("fill", "#fff").attr("stroke", "#000").attr("stroke-width", 2.5)
    .attr("transform", `translate(${km.x(vx)},${km.y(vy)})`);
  g.append("text").attr("x", km.x(vx) + 14).attr("y", km.y(vy) - 10).attr("font-weight", 700).attr("font-size", 12)
    .attr("stroke", "#fff").attr("stroke-width", 3).attr("paint-order", "stroke").text("Your field");
  $("#kfield").innerHTML = `★ Your field belongs to <b style="color:${ZPAL[best % ZPAL.length]}">cluster ${best + 1}</b> (live as you move the sliders)`;
}
const kmRestart = () => { kmReset(); kmPlay(); };
$("#kk").addEventListener("input", kmRestart); $("#kfx").addEventListener("change", kmRestart); $("#kfy").addEventListener("change", kmRestart);
$("#kplay").onclick = () => KM.timer ? kmStop() : kmPlay();
$("#kstep").onclick = () => { kmStop(); kmStep(); };
$("#kreset").onclick = () => { kmReset(); kmPlay(); };
$("#kspeed").addEventListener("input", () => { if (KM.timer) kmPlay(); });

/* ---------------- 4. live regression ---------------- */
function scatterCurve(sel, pts, curve, marker, xl, color) {
  const W = 560, H = 360, m = { l: 52, r: 14, t: 14, b: 44 };
  const svg = d3.select(sel); svg.selectAll("*").remove(); svg.attr("viewBox", `0 0 ${W} ${H}`);
  const ex = d3.extent(pts.concat(curve, [marker]), d => d.x);
  const x = d3.scaleLinear(ex, [m.l, W - m.r]).nice();
  const y = d3.scaleLinear([0, d3.max(pts.concat(curve, [marker]), d => d.y)], [H - m.b, m.t]).nice();
  svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x));
  svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y));
  svg.append("text").attr("x", W / 2).attr("y", H - 8).attr("text-anchor", "middle").attr("font-size", 12).text(xl);
  svg.append("text").attr("transform", "rotate(-90)").attr("x", -H / 2).attr("y", 14).attr("text-anchor", "middle").attr("font-size", 12).text("Yield (t/ha)");
  svg.selectAll("circle").data(pts).join("circle").attr("r", 2.6).attr("fill", color).attr("fill-opacity", .3).attr("cx", d => x(d.x)).attr("cy", d => y(d.y));
  svg.append("path").datum(curve).attr("fill", "none").attr("stroke", "#000").attr("stroke-width", 3).attr("d", d3.line().x(d => x(d.x)).y(d => y(d.y)).curve(d3.curveMonotoneX));
  svg.append("line").attr("x1", x(marker.x)).attr("x2", x(marker.x)).attr("y1", m.t).attr("y2", H - m.b).attr("stroke", "#d32f2f").attr("stroke-dasharray", "5 4");
  svg.append("circle").attr("cx", x(marker.x)).attr("cy", y(marker.y)).attr("r", 8).attr("fill", "#d32f2f").attr("stroke", "#fff").attr("stroke-width", 2);
  svg.append("text").attr("x", Math.min(x(marker.x) + 12, W - 110)).attr("y", Math.max(y(marker.y) - 12, 24)).attr("font-weight", 700).attr("fill", "#d32f2f")
    .attr("stroke", "#fff").attr("stroke-width", 3).attr("paint-order", "stroke").text(`${marker.y.toFixed(2)} t/ha`);
}
let regToken = 0;
async function loadReg() {
  const crop = $("#rcrop").value, tok = ++regToken;
  const q = new URLSearchParams({ crop, soil: S.soil, rain: S.rain, temp: S.temp });
  const d = await (await fetch("/api/regression?" + q)).json();
  if (tok !== regToken) return;
  const col = CFG.colors[crop];
  scatterCurve("#reg1", d.pts.map(p => ({ x: p.r, y: p.y })), d.rain_curve, { x: S.rain, y: d.pred }, `Rainfall (mm) - at ${S.temp} °C, ${S.soil} soil`, col);
  scatterCurve("#reg2", d.pts.map(p => ({ x: p.t, y: p.y })), d.temp_curve, { x: S.temp, y: d.pred }, `Temperature (°C) - at ${S.rain} mm, ${S.soil} soil`, col);
  $("#regpred").innerHTML = `Predicted <b>${crop}</b> yield: <b>${d.pred} t/ha</b> <span class="muted">(crop average ${d.avg} t/ha)</span>`;
}
async function loadTrend() {
  const crop = $("#rcrop").value, d = await (await fetch("/api/trend?crop=" + crop)).json();
  const W = 560, H = 260, m = { l: 52, r: 14, t: 14, b: 36 };
  const svg = d3.select("#trend"); svg.selectAll("*").remove(); svg.attr("viewBox", `0 0 ${W} ${H}`);
  const x = d3.scaleLinear(d3.extent(d, e => e.year), [m.l, W - m.r]);
  const y = d3.scaleLinear([d3.min(d, e => e.value) * .95, d3.max(d, e => e.value) * 1.05], [H - m.b, m.t]);
  svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).tickFormat(d3.format("d")).ticks(8));
  svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(6));
  svg.append("path").datum(d).attr("fill", "none").attr("stroke", CFG.colors[crop]).attr("stroke-width", 2.5).attr("d", d3.line().x(e => x(e.year)).y(e => y(e.value)));
  svg.selectAll("circle").data(d).join("circle").attr("r", 4).attr("fill", CFG.colors[crop]).attr("cx", e => x(e.year)).attr("cy", e => y(e.value))
    .append("title").text(e => `${e.year}: ${e.value.toFixed(2)} t/ha`);
}
$("#rcrop").addEventListener("change", () => { window.regTouched = true; loadReg(); loadTrend(); });

/* ---------------- 5. live data explorer (zoom + hover + legend filter) ---------------- */
const EX = { hidden: new Set(), W: 760, H: 480, m: { l: 58, r: 16, t: 14, b: 48 } };
function exDraw() {
  const fx = $("#efx").value, fy = $("#efy").value, svg = d3.select("#exsvg");
  svg.selectAll("*").remove(); svg.attr("viewBox", `0 0 ${EX.W} ${EX.H}`);
  const x0 = d3.scaleLinear(d3.extent(SAMPLE, p => p[fx]), [EX.m.l, EX.W - EX.m.r]).nice();
  const y0 = d3.scaleLinear(d3.extent(SAMPLE, p => p[fy]), [EX.H - EX.m.b, EX.m.t]).nice();
  svg.append("defs").append("clipPath").attr("id", "exclip").append("rect").attr("x", EX.m.l).attr("y", EX.m.t)
    .attr("width", EX.W - EX.m.l - EX.m.r).attr("height", EX.H - EX.m.t - EX.m.b);
  const gx = svg.append("g").attr("transform", `translate(0,${EX.H - EX.m.b})`), gy = svg.append("g").attr("transform", `translate(${EX.m.l},0)`);
  svg.append("text").attr("x", EX.W / 2).attr("y", EX.H - 8).attr("text-anchor", "middle").attr("font-size", 12).text(FEATS[fx]);
  svg.append("text").attr("transform", "rotate(-90)").attr("x", -EX.H / 2).attr("y", 14).attr("text-anchor", "middle").attr("font-size", 12).text(FEATS[fy]);
  const pts = svg.append("g").attr("clip-path", "url(#exclip)");
  const vis = () => SAMPLE.filter(p => !EX.hidden.has(p.c));
  function paint(x, y) {
    gx.call(d3.axisBottom(x)); gy.call(d3.axisLeft(y));
    pts.selectAll("circle").data(vis(), p => p).join("circle").attr("r", 3.4).attr("fill", p => CFG.colors[p.c]).attr("fill-opacity", .6)
      .attr("cx", p => x(p[fx])).attr("cy", p => y(p[fy]))
      .on("mouseenter", (e, p) => { const t = $("#tip"); t.style.display = "block"; t.innerHTML = `<b>${p.c}</b> · ${p.s} soil<br>Rain ${p.r.toFixed(0)} mm · Temp ${p.t.toFixed(1)} °C<br>pH ${p.p.toFixed(1)} · Humidity ${p.h.toFixed(0)}%<br>Yield <b>${p.y.toFixed(2)} t/ha</b> · ${p.yr}`; })
      .on("mousemove", e => { const t = $("#tip"); t.style.left = (e.pageX + 14) + "px"; t.style.top = (e.pageY + 14) + "px"; })
      .on("mouseleave", () => { $("#tip").style.display = "none"; });
  }
  paint(x0, y0);
  svg.call(d3.zoom().scaleExtent([1, 30]).translateExtent([[0, 0], [EX.W, EX.H]]).extent([[0, 0], [EX.W, EX.H]])
    .on("zoom", e => paint(e.transform.rescaleX(x0), e.transform.rescaleY(y0))));
}
function exLegend() {
  const L = $("#elegend"); L.innerHTML = "";
  CFG.crops.forEach(c => {
    const b = document.createElement("span"); b.className = "chip" + (EX.hidden.has(c) ? " off" : "");
    b.innerHTML = `<i style="background:${CFG.colors[c]}"></i>${c}`;
    b.onclick = () => { EX.hidden.has(c) ? EX.hidden.delete(c) : EX.hidden.add(c); exLegend(); exDraw(); };
    L.appendChild(b);
  });
}
$("#efx").addEventListener("change", exDraw); $("#efy").addEventListener("change", exDraw);

/* ---------------- start ---------------- */
(async function init() {
  readControls();
  SAMPLE = await (await fetch("/api/sample")).json();
  exLegend(); exDraw();
  await loadTree();
  kmReset(); kmPlay();
  refreshAll(); loadTrend();
})();
