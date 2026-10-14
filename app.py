"""
Smart Agriculture Crop Recommendation  -  Data Mining Web App
Algorithms : Decision Tree (classification), Regression (yield prediction), K-Means (clustering)
Run        : python app.py   ->  open http://127.0.0.1:5000
"""
import io, os, base64
import numpy as np, pandas as pd
from flask import Flask, render_template, request, jsonify, Response, abort
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, confusion_matrix, r2_score,
                             mean_absolute_error, mean_squared_error)

BASE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(BASE, "data", "crop_data.csv")
if not os.path.exists(CSV):                       # auto-generate if missing
    import subprocess, sys
    subprocess.check_call([sys.executable, os.path.join(BASE, "generate_dataset.py")], cwd=BASE)

app = Flask(__name__)
df = pd.read_csv(CSV)
SOILS = sorted(df.soil_type.unique())
CROPS = sorted(df.crop.unique())
PALETTE = ["#2e7d32","#f9a825","#1565c0","#c62828","#6a1b9a","#00838f",
           "#ef6c00","#558b2f","#ad1457","#4e342e","#455a64","#9e9d24"]
CROP_COLOR = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(CROPS)}

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def X_class(frame):
    """Feature matrix: rainfall, temperature, one-hot soil."""
    X = frame[["rainfall_mm", "avg_temp_c"]].copy()
    for s in SOILS:
        X["soil_" + s] = (frame["soil_type"] == s).astype(int)
    return X

# --------------------------------------------------------------------------
# 1) DECISION TREE  - crop classification
# --------------------------------------------------------------------------
Xc = X_class(df); yc = df["crop"]
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.2, random_state=1, stratify=yc)
tree = DecisionTreeClassifier(max_depth=10, min_samples_leaf=25, random_state=1).fit(Xc_tr, yc_tr)
yc_pred = tree.predict(Xc_te)
TREE_ACC = accuracy_score(yc_te, yc_pred)

# --------------------------------------------------------------------------
# 2) REGRESSION - yield prediction (one polynomial regression per crop)
# --------------------------------------------------------------------------
reg_models, reg_test_true, reg_test_pred, reg_test_crop = {}, [], [], []
for crop in CROPS:
    d = df[df.crop == crop]
    Xr = X_class(d); yr = d["yield_t_ha"]
    a, b, c, e = train_test_split(Xr, yr, test_size=0.2, random_state=1)
    m = make_pipeline(StandardScaler(), PolynomialFeatures(3), LinearRegression()).fit(a, c)
    reg_models[crop] = m
    p = m.predict(b)
    reg_test_true += list(e); reg_test_pred += list(p); reg_test_crop += [crop]*len(e)
reg_test_true = np.array(reg_test_true); reg_test_pred = np.array(reg_test_pred)
reg_test_crop = np.array(reg_test_crop)
REG_R2 = r2_score(reg_test_true, reg_test_pred)
REG_MAE = mean_absolute_error(reg_test_true, reg_test_pred)
REG_RMSE = float(np.sqrt(mean_squared_error(reg_test_true, reg_test_pred)))
# R2 on a normalised scale (yield / crop mean) so sugarcane/potato do not dominate
_norm = df.groupby("crop")["yield_t_ha"].mean()
_scale = np.array([_norm[c] for c in reg_test_crop])
REG_R2_NORM = r2_score(reg_test_true/_scale, reg_test_pred/_scale)

YMIN = df.groupby("crop")["yield_t_ha"].min().to_dict()
YMAX = df.groupby("crop")["yield_t_ha"].max().to_dict()

def predict_yield(crop, rain, temp, soil):
    """Regression prediction, clipped to the yield range actually observed for that crop."""
    row = pd.DataFrame([{"rainfall_mm": rain, "avg_temp_c": temp, "soil_type": soil}])
    return float(np.clip(reg_models[crop].predict(X_class(row))[0], YMIN[crop], YMAX[crop]))

# --------------------------------------------------------------------------
# 3) K-MEANS - agro-climatic zones
# --------------------------------------------------------------------------
KM_FEATS = ["rainfall_mm", "avg_temp_c", "soil_ph"]
km_scaler = StandardScaler().fit(df[KM_FEATS])
Xk = km_scaler.transform(df[KM_FEATS])
K = 5
kmeans = KMeans(n_clusters=K, n_init=10, random_state=1).fit(Xk)
df["cluster"] = kmeans.labels_
INERTIA = [KMeans(n_clusters=k, n_init=5, random_state=1).fit(Xk).inertia_ for k in range(1, 11)]

def name_cluster(r, t, p):
    rain = "Dry" if r < 600 else ("Moderate-rain" if r < 1200 else "Wet")
    temp = "Cool" if t < 20 else ("Warm" if t < 26 else "Hot")
    soil = "alkaline soil" if p > 7.3 else ("acidic soil" if p < 6.5 else "neutral soil")
    return f"{rain}, {temp}, {soil}"
CLUSTER_INFO = {}
for k in range(K):
    d = df[df.cluster == k]
    top = d.crop.value_counts().head(3)
    CLUSTER_INFO[k] = dict(
        name=name_cluster(d.rainfall_mm.mean(), d.avg_temp_c.mean(), d.soil_ph.mean()),
        size=len(d), rain=int(round(d.rainfall_mm.mean())), temp=round(float(d.avg_temp_c.mean()), 1),
        ph=round(float(d.soil_ph.mean()), 2), top=", ".join(f"{c} ({n})" for c, n in top.items()))
for k, v in CLUSTER_INFO.items():
    v["label"] = f"Zone {k+1}: {v['name']}"

# --------------------------------------------------------------------------
# plotting utils (pure matplotlib Figure API -> thread safe, works offline)
# --------------------------------------------------------------------------
def png(fig):
    fig.tight_layout()
    buf = io.BytesIO(); FigureCanvasAgg(fig).print_png(buf)
    return buf.getvalue()

def resp(fig):
    return Response(png(fig), mimetype="image/png", headers={"Cache-Control": "no-cache"})

def style(ax, title, xl="", yl=""):
    ax.set_title(title, fontsize=12, fontweight="bold", color="#1b5e20")
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.grid(alpha=.25)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

def plot_crop_distribution():
    fig = Figure(figsize=(8, 4.6)); ax = fig.subplots()
    vc = df.crop.value_counts()
    ax.bar(vc.index, vc.values, color=[CROP_COLOR[c] for c in vc.index])
    style(ax, "Records per crop", "", "Number of records")
    ax.tick_params(axis="x", rotation=40)
    return fig

def plot_correlation():
    cols = ["rainfall_mm", "avg_temp_c", "soil_ph", "humidity_pct", "yield_t_ha"]
    cm = df[cols].corr().values
    fig = Figure(figsize=(6.4, 5.2)); ax = fig.subplots()
    im = ax.imshow(cm, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=35, ha="right")
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=.046)
    ax.set_title("Correlation heat-map", fontsize=12, fontweight="bold", color="#1b5e20")
    return fig

def plot_rain_temp_scatter():
    s = df.sample(4000, random_state=3)
    fig = Figure(figsize=(8, 5.2)); ax = fig.subplots()
    for c in CROPS:
        d = s[s.crop == c]
        ax.scatter(d.rainfall_mm, d.avg_temp_c, s=8, alpha=.6, color=CROP_COLOR[c], label=c)
    style(ax, "Crop patterns: rainfall vs temperature", "Rainfall (mm)", "Average temperature (°C)")
    ax.legend(ncol=2, fontsize=8, markerscale=2.2)
    return fig

def plot_yield_box():
    fig = Figure(figsize=(8, 4.8)); ax = fig.subplots()
    order = df.groupby("crop")["yield_t_ha"].median().sort_values().index
    # sugarcane / potato have much larger yields -> use log scale for readability
    bp = ax.boxplot([df[df.crop == c].yield_t_ha for c in order], tick_labels=order,
                    patch_artist=True, showfliers=False)
    for p, c in zip(bp["boxes"], order): p.set_facecolor(CROP_COLOR[c]); p.set_alpha(.75)
    ax.set_yscale("log")
    style(ax, "Yield distribution by crop (log scale)", "", "Yield (t/ha)")
    ax.tick_params(axis="x", rotation=40)
    return fig

def plot_tree_fig():
    fig = Figure(figsize=(20, 9)); FigureCanvasAgg(fig); ax = fig.subplots()
    ann = plot_tree(tree, max_depth=3, feature_names=list(Xc.columns), class_names=list(tree.classes_),
              filled=True, rounded=True, fontsize=11, ax=ax, impurity=False, proportion=True, label="none")
    for t in ann:                      # hide the long per-class value arrays for readability
        txt = t.get_text()
        if "[" in txt:                 # drop the multi-line class-proportion array
            txt = txt[:txt.index("[")] + txt[txt.index("]")+1:].lstrip("\n")
        t.set_text("\n".join(l for l in txt.split("\n") if l.strip()))
    ax.set_title(f"Decision Tree (top 3 levels shown of depth {tree.get_depth()}) - accuracy {TREE_ACC*100:.1f}%",
                 fontsize=14, fontweight="bold", color="#1b5e20")
    return fig

def plot_importance():
    imp = pd.Series(tree.feature_importances_, index=Xc.columns).sort_values()
    fig = Figure(figsize=(7, 4.4)); ax = fig.subplots()
    ax.barh(imp.index, imp.values, color="#2e7d32")
    style(ax, "Decision-tree feature importance", "Importance", "")
    return fig

def plot_confusion():
    cm = confusion_matrix(yc_te, yc_pred, labels=tree.classes_)
    cm = cm / cm.sum(axis=1, keepdims=True)
    fig = Figure(figsize=(7.2, 6)); ax = fig.subplots()
    im = ax.imshow(cm, cmap="Greens", vmin=0, vmax=1)
    n = len(tree.classes_)
    ax.set_xticks(range(n)); ax.set_xticklabels(tree.classes_, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(tree.classes_, fontsize=8)
    for i in range(n):
        for j in range(n):
            if cm[i, j] >= .05:
                ax.text(j, i, f"{cm[i,j]*100:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > .5 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (% of each actual crop)", fontsize=12, fontweight="bold", color="#1b5e20")
    fig.colorbar(im, ax=ax, fraction=.046)
    return fig

def plot_reg_actual_pred():
    fig = Figure(figsize=(7, 5.2)); ax = fig.subplots()
    # normalised by crop mean so all crops share one axis
    for c in CROPS:
        m = reg_test_crop == c
        ax.scatter(reg_test_true[m]/_norm[c], reg_test_pred[m]/_norm[c], s=5, alpha=.35, color=CROP_COLOR[c], label=c)
    lim = [0, float((reg_test_true/_scale).max())]
    ax.plot(lim, lim, "k--", lw=1)
    style(ax, f"Regression: actual vs predicted yield (R²={REG_R2:.3f})",
          "Actual yield (relative to crop average)", "Predicted yield (relative to crop average)")
    ax.legend(ncol=3, fontsize=7, markerscale=3)
    return fig

def plot_reg_curve(crop, soil):
    d = df[(df.crop == crop)]
    fig = Figure(figsize=(8, 5)); ax = fig.subplots()
    ax.scatter(d.rainfall_mm, d.yield_t_ha, s=6, alpha=.25, color=CROP_COLOR[crop], label="Historical data")
    rs = np.linspace(d.rainfall_mm.min(), d.rainfall_mm.max(), 120)
    t = d.avg_temp_c.mean()
    ys = [predict_yield(crop, r, t, soil) for r in rs]
    ax.plot(rs, ys, color="black", lw=2.2, label=f"Regression curve ({soil} soil, {t:.1f}°C)")
    style(ax, f"{crop}: rainfall vs yield", "Rainfall (mm)", "Yield (t/ha)")
    ax.legend(fontsize=9)
    return fig

def plot_year_trend(crop):
    d = df[df.crop == crop].groupby("year")["yield_t_ha"].mean()
    fig = Figure(figsize=(8, 4.2)); ax = fig.subplots()
    ax.plot(d.index, d.values, marker="o", color=CROP_COLOR[crop])
    style(ax, f"{crop}: average yield by year", "Year", "Yield (t/ha)")
    return fig

def plot_elbow():
    fig = Figure(figsize=(6.6, 4.6)); ax = fig.subplots()
    ax.plot(range(1, 11), INERTIA, marker="o", color="#2e7d32")
    ax.axvline(K, color="#c62828", ls="--"); ax.text(K+.15, max(INERTIA)*.8, f"chosen k={K}", color="#c62828")
    style(ax, "K-Means elbow curve", "Number of clusters (k)", "Inertia")
    return fig

def plot_clusters():
    s = df.sample(5000, random_state=5)
    fig = Figure(figsize=(8, 5.4)); ax = fig.subplots()
    cols = PALETTE
    for k in range(K):
        d = s[s.cluster == k]
        ax.scatter(d.rainfall_mm, d.avg_temp_c, s=8, alpha=.55, color=cols[k], label=CLUSTER_INFO[k]["label"])
    cen = km_scaler.inverse_transform(kmeans.cluster_centers_)
    ax.scatter(cen[:, 0], cen[:, 1], marker="X", s=220, color="black", edgecolor="white", label="Centroids")
    style(ax, "K-Means agro-climatic zones", "Rainfall (mm)", "Average temperature (°C)")
    ax.legend(fontsize=8)
    return fig

def plot_cluster_crops():
    ct = pd.crosstab(df.cluster, df.crop, normalize="index") * 100
    ct.index = [f"Zone {i+1}" for i in ct.index]
    fig = Figure(figsize=(8, 4.8)); ax = fig.subplots()
    bottom = np.zeros(len(ct))
    for c in ct.columns:
        ax.bar(ct.index, ct[c].values, bottom=bottom, color=CROP_COLOR[c], label=c)
        bottom += ct[c].values
    style(ax, "Crop mix inside each K-Means zone", "", "% of records")
    ax.legend(fontsize=7, ncol=2, bbox_to_anchor=(1.01, 1), loc="upper left")
    return fig

PLOTS = {
    "crop_distribution": lambda a: plot_crop_distribution(),
    "correlation": lambda a: plot_correlation(),
    "scatter": lambda a: plot_rain_temp_scatter(),
    "yield_box": lambda a: plot_yield_box(),
    "tree": lambda a: plot_tree_fig(),
    "importance": lambda a: plot_importance(),
    "confusion": lambda a: plot_confusion(),
    "reg_actual_pred": lambda a: plot_reg_actual_pred(),
    "reg_curve": lambda a: plot_reg_curve(a["crop"], a["soil"]),
    "year_trend": lambda a: plot_year_trend(a["crop"]),
    "elbow": lambda a: plot_elbow(),
    "clusters": lambda a: plot_clusters(),
    "cluster_crops": lambda a: plot_cluster_crops(),
}

# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------
@app.route("/analysis")
def analysis():
    stats = df[["rainfall_mm", "avg_temp_c", "soil_ph", "humidity_pct", "yield_t_ha"]].describe().round(2)
    return render_template("analysis.html", page="analysis", crops=CROPS, soils=SOILS,
        n=len(df), ncrops=len(CROPS), nsoils=len(SOILS), years=f"{df.year.min()}-{df.year.max()}",
        tree_acc=round(TREE_ACC*100, 2), depth=tree.get_depth(), leaves=tree.get_n_leaves(),
        r2=round(REG_R2, 4), r2n=round(REG_R2_NORM, 4), mae=round(REG_MAE, 3), rmse=round(REG_RMSE, 3),
        clusters=CLUSTER_INFO, k=K,
        stats_html=stats.to_html(classes="tbl", border=0),
        sample_html=df.sample(12, random_state=7).drop(columns="cluster").to_html(classes="tbl", index=False, border=0))

@app.route("/plot/<name>.png")
def plot(name):
    if name not in PLOTS: abort(404)
    crop = request.args.get("crop", CROPS[0]); soil = request.args.get("soil", SOILS[0])
    if crop not in CROPS or soil not in SOILS: abort(400)
    return resp(PLOTS[name]({"crop": crop, "soil": soil}))

@app.route("/api/recommend", methods=["POST"])
def recommend():
    j = request.get_json(force=True)
    try:
        rain = float(j["rainfall"]); temp = float(j["temperature"]); soil = j["soil"]
    except Exception:
        return jsonify(error="Invalid input"), 400
    if soil not in SOILS: return jsonify(error="Unknown soil"), 400
    row = pd.DataFrame([{"rainfall_mm": rain, "avg_temp_c": temp, "soil_type": soil}])
    proba = tree.predict_proba(X_class(row))[0]
    order = np.argsort(proba)[::-1]
    results = []
    for i in order[:5]:
        crop = tree.classes_[i]
        results.append(dict(crop=crop, probability=round(float(proba[i])*100, 1),
                            yield_t_ha=round(predict_yield(crop, rain, temp, soil), 2)))
    # make sure at least 3 results are shown even when tree is very sure
    # K-Means zone of this input (soil pH = typical pH of chosen soil)
    ph = float(df[df.soil_type == soil].soil_ph.mean())
    z = int(kmeans.predict(km_scaler.transform(pd.DataFrame([[rain, temp, ph]], columns=KM_FEATS)))[0])
    # chart as base64 png
    fig = Figure(figsize=(6.4, 3.4)); ax = fig.subplots()
    names = [r["crop"] for r in results][::-1]; vals = [r["probability"] for r in results][::-1]
    ax.barh(names, vals, color=[CROP_COLOR[n] for n in names])
    for y, v in enumerate(vals): ax.text(v+1, y, f"{v}%", va="center", fontsize=9)
    ax.set_xlim(0, 115); style(ax, "Suitability (decision-tree probability)", "%", "")
    img = base64.b64encode(png(fig)).decode()
    return jsonify(best=results[0], results=results, chart=img,
                   zone=CLUSTER_INFO[z]["label"], zone_top=CLUSTER_INFO[z]["top"])

# --------------------------------------------------------------------------
# LIVE (interactive) APIs
# --------------------------------------------------------------------------
_s = df.sample(3000, random_state=11)
SAMPLE = [dict(r=float(a), t=float(b), p=float(c), h=float(d), y=float(e), c=f, s=g, yr=int(h))
          for a, b, c, d, e, f, g, h in zip(_s.rainfall_mm, _s.avg_temp_c, _s.soil_ph, _s.humidity_pct,
                                            _s.yield_t_ha, _s.crop, _s.soil_type, _s.year)]
AVG_YIELD = df.groupby("crop")["yield_t_ha"].mean().round(3).to_dict()
PH_BY_SOIL = df.groupby("soil_type")["soil_ph"].mean().round(2).to_dict()

def input_row(rain, temp, soil):
    return pd.DataFrame([{"rainfall_mm": rain, "avg_temp_c": temp, "soil_type": soil}])

def predict_yield_many(crop, rains, temps, soil):
    fr = pd.DataFrame({"rainfall_mm": rains, "avg_temp_c": temps, "soil_type": soil})
    return np.clip(reg_models[crop].predict(X_class(fr)), YMIN[crop], YMAX[crop])

def get_args():
    try:
        rain = float(request.args.get("rain", 800)); temp = float(request.args.get("temp", 25))
    except ValueError:
        abort(400)
    soil = request.args.get("soil", SOILS[0])
    if soil not in SOILS: abort(400)
    return rain, temp, soil

@app.route("/favicon.ico")
def favicon():
    return Response(status=204)

@app.route("/")
@app.route("/live")
def live():
    cfg = dict(crops=CROPS, soils=SOILS, colors=CROP_COLOR, phBySoil=PH_BY_SOIL, avg=AVG_YIELD,
               rmin=int(df.rainfall_mm.min()), rmax=int(df.rainfall_mm.max()),
               tmin=int(df.avg_temp_c.min()), tmax=int(df.avg_temp_c.max()))
    return render_template("live.html", page="home", cfg=cfg, soils=SOILS, crops=CROPS,
                           rmin=cfg["rmin"], rmax=cfg["rmax"], tmin=cfg["tmin"], tmax=cfg["tmax"])

@app.route("/api/sample")
def api_sample():
    return jsonify(SAMPLE)

@app.route("/api/all_crops")
def api_all_crops():
    rain, temp, soil = get_args()
    proba = tree.predict_proba(X_class(input_row(rain, temp, soil)))[0]
    out = []
    for c, p in zip(tree.classes_, proba):
        y = predict_yield(c, rain, temp, soil)
        out.append(dict(crop=c, prob=round(float(p)*100, 1), yield_t_ha=round(y, 2),
                        index=round(y / AVG_YIELD[c] * 100, 0)))
    return jsonify(out)

_tree_cache = {}
def get_tree(depth, leaf):
    key = (depth, leaf)
    if key in _tree_cache: return _tree_cache[key]
    m = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=1).fit(Xc_tr, yc_tr)
    t = m.tree_; classes = m.classes_; cols = list(Xc.columns); counter = [0]
    def build(i):
        nid = counter[0]; counter[0] += 1
        v = t.value[i][0]; v = v / v.sum()
        top = np.argsort(v)[::-1][:3]
        node = dict(id=nid, n=int(t.n_node_samples[i]), cls=str(classes[top[0]]), p=round(float(v[top[0]]), 3),
                    top=[[str(classes[j]), round(float(v[j]), 3)] for j in top if v[j] > 0.005])
        if t.children_left[i] != -1:
            node["f"] = cols[t.feature[i]]; node["th"] = round(float(t.threshold[i]), 3)
            node["ch"] = [build(t.children_left[i]), build(t.children_right[i])]
        return node
    res = dict(tree=build(0), depth=int(m.get_depth()), leaves=int(m.get_n_leaves()),
               nodes=int(t.node_count),
               acc_test=round(float(accuracy_score(yc_te, m.predict(Xc_te)))*100, 2),
               acc_train=round(float(accuracy_score(yc_tr, m.predict(Xc_tr)))*100, 2),
               importance={c: round(float(i), 4) for c, i in zip(cols, m.feature_importances_)})
    if len(_tree_cache) > 40: _tree_cache.clear()
    _tree_cache[key] = res
    return res

@app.route("/api/tree")
def api_tree():
    try:
        depth = max(1, min(14, int(request.args.get("depth", 4))))
        leaf = max(1, min(500, int(request.args.get("leaf", 25))))
    except ValueError:
        abort(400)
    return jsonify(get_tree(depth, leaf))

@app.route("/api/regression")
def api_regression():
    rain, temp, soil = get_args()
    crop = request.args.get("crop", CROPS[0])
    if crop not in CROPS: abort(400)
    d = df[df.crop == crop]
    pts = d.sample(min(700, len(d)), random_state=2)
    rs = np.linspace(d.rainfall_mm.min(), d.rainfall_mm.max(), 70)
    ts = np.linspace(d.avg_temp_c.min(), d.avg_temp_c.max(), 70)
    yr = predict_yield_many(crop, rs, np.full_like(rs, temp), soil)
    yt = predict_yield_many(crop, np.full_like(ts, rain), ts, soil)
    return jsonify(
        pts=[dict(r=float(a), t=float(b), y=float(c)) for a, b, c in zip(pts.rainfall_mm, pts.avg_temp_c, pts.yield_t_ha)],
        rain_curve=[dict(x=float(a), y=float(b)) for a, b in zip(rs, yr)],
        temp_curve=[dict(x=float(a), y=float(b)) for a, b in zip(ts, yt)],
        pred=round(predict_yield(crop, rain, temp, soil), 2), avg=round(AVG_YIELD[crop], 2))

@app.route("/api/trend")
def api_trend():
    crop = request.args.get("crop", CROPS[0])
    if crop not in CROPS: abort(400)
    g = df[df.crop == crop].groupby("year")["yield_t_ha"].mean()
    return jsonify([dict(year=int(y), value=float(v)) for y, v in g.items()])

if __name__ == "__main__":
    print(f"Dataset rows: {len(df)} | Decision-tree accuracy: {TREE_ACC*100:.1f}% | Regression R2: {REG_R2:.3f}")
    print("Open http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
