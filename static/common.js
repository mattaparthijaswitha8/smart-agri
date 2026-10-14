/* Shared d3 helpers (works offline - d3 is bundled in /static/d3.min.js) */
function barChart(sel, data, o) {
  o = o || {};
  const svg = d3.select(sel), W = o.width || 560, rowH = o.rowH || 28, pad = 8;
  const left = o.left || 100, right = o.right || 80;
  svg.attr("viewBox", `0 0 ${W} ${data.length * rowH + pad * 2}`);
  const max = o.max || d3.max(data, d => d.value) || 1;
  const x = d3.scaleLinear([0, max], [0, W - left - right]);
  const rows = svg.selectAll("g.row").data(data, d => d.key);
  const ent = rows.enter().append("g").attr("class", "row")
      .attr("transform", (d, i) => `translate(0,${pad + i * rowH})`);
  ent.append("text").attr("class", "lab").attr("x", left - 8).attr("y", rowH / 2)
      .attr("text-anchor", "end").attr("dominant-baseline", "middle").attr("font-size", 12);
  ent.append("rect").attr("class", "bar").attr("x", left).attr("y", 3).attr("height", rowH - 8).attr("rx", 4).attr("width", 0);
  ent.append("text").attr("class", "val").attr("y", rowH / 2).attr("x", left + 6)
      .attr("dominant-baseline", "middle").attr("font-size", 12).attr("font-weight", 700).attr("fill", "#1b5e20");
  const all = ent.merge(rows);
  all.transition().duration(350).attr("transform", (d, i) => `translate(0,${pad + i * rowH})`);
  all.select(".lab").text(d => d.label);
  all.select("rect.bar").attr("fill", d => d.color).transition().duration(350).attr("width", d => Math.max(1, x(d.value)));
  all.select(".val").text(d => d.text).transition().duration(350).attr("x", d => left + x(d.value) + 6);
  rows.exit().remove();
}
