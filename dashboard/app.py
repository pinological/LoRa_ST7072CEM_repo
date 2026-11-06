"""
LoRa Hill-Terrain Explorer, a Gradio dashboard for the ST7072CEM paper
"Experimental Characterization of LoRa Propagation and Machine Learning-Based
Path Loss Modeling Across Diverse Hill Terrains in Nepal" (Peter Karki).

Reads the raw Roamer logs from LoRa_data/ (next to app.py or one folder up).
Research prototype: no coordinates or site names are shown or exported.
"""
import os
import time
import tempfile
from functools import lru_cache

import numpy as np
import pandas as pd
import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import map_coordinates, gaussian_filter
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, KFold, cross_val_predict
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score, roc_curve, confusion_matrix,
                             precision_score, recall_score, f1_score)

from loader import find_data_dir, load_data

# ------------------------------------------------------------------ data
HERE = os.path.dirname(os.path.abspath(__file__))
try:
    DATA_DIR = find_data_dir()
    PK, STOPS, LOST = load_data(DATA_DIR)
    DATA_SOURCE = f"raw logs in {DATA_DIR.name}/"
except FileNotFoundError:
    D = os.path.join(HERE, "data")
    STOPS = pd.read_csv(os.path.join(D, "stops.csv"))
    PK = pd.read_csv(os.path.join(D, "packets.csv"))
    LOST = pd.read_csv(os.path.join(D, "lost_locations.csv"))
    DATA_SOURCE = "anonymised tables in data/"
print("Data source:", DATA_SOURCE, "|", len(PK), "packets,", len(STOPS), "stops,", len(LOST), "lost locations")

SFS, PWRS = [7, 8, 9, 10, 11, 12], [2, 8, 14]
LCS = ["forest", "built-up", "mixed", "open"]
CASE_NAME = {1: "Case 1 · forested slope", 2: "Case 2 · hill slope", 3: "Case 3 · rooftops"}
CASE_COL = {1: "#3B8FD9", 2: "#F0A73A", 3: "#21B38A"}
PWR_COL = {2: "#9AA5B1", 8: "#3B8FD9", 14: "#E4572E"}
LC_COL = {"forest": "#2E8B57", "built-up": "#9B59B6", "mixed": "#F0A73A", "open": "#3B8FD9"}

for df in (PK, STOPS):
    df["logd"] = np.log10(df.distance_to_station_m)
for c in ["built-up", "mixed", "open"]:
    PK["lc_" + c] = (PK.land_cover == c).astype(int)
LC = ["lc_built-up", "lc_mixed", "lc_open"]
FEATURE_SETS = {
    "SF + power": ["sf", "pwr"],
    "+ distance": ["sf", "pwr", "logd"],
    "+ elevation difference and angle": ["sf", "pwr", "logd", "elev_diff_m", "elev_angle_deg"],
    "+ land cover (all features)": ["sf", "pwr", "logd", "elev_diff_m", "elev_angle_deg"] + LC,
}
ALL = FEATURE_SETS["+ land cover (all features)"]
COST = lambda sf, p: 10 ** (p / 10) * 2 ** (sf - 7)
FULL_COST = COST(12, 14)
COSTS = sorted([(COST(s, p), s, p) for s in SFS for p in PWRS])


def stop_sort(k):
    return (k[0], int("".join(ch for ch in k[1:] if ch.isdigit()) or 0))


grid = PK.groupby(["stop_key", "sf", "pwr"]).ok.sum().reset_index()
grid["cost"] = COST(grid.sf, grid.pwr)
best = grid[grid.ok >= 4].sort_values("cost").groupby("stop_key").first()
STOPS = STOPS.merge(best[["sf", "pwr", "cost"]].rename(columns={"sf": "best_sf", "pwr": "best_pwr", "cost": "best_cost"}),
                    left_on="stop_key", right_index=True, how="left")
STOPS["saving_pct"] = 100 * (1 - STOPS.best_cost / FULL_COST)

# ------------------------------------------------------------------ plot style (readable in dark and light mode)
FONT = "#8a96a6"
GRID = "rgba(138,150,166,0.22)"


def style(fig, title=None, h=420, legend_bottom=True):
    fig.update_layout(
        title=dict(text=title, x=0.01, font=dict(size=15)) if title else None,
        height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(138,150,166,0.06)",
        font=dict(color=FONT, family="Inter, Segoe UI, sans-serif", size=12),
        margin=dict(t=56 if title else 24, l=60, r=24, b=60),
        hoverlabel=dict(bgcolor="#1f2937", font_color="#f9fafb", bordercolor="#1f2937"),
        legend=dict(orientation="h", y=-0.2, x=0, bgcolor="rgba(0,0,0,0)") if legend_bottom else dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_layout(autosize=True)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return fig


def empty(msg="No stops match the filters"):
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=15, color=FONT), x=.5, y=.5, xref="paper", yref="paper")
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    return style(fig, h=320)


# ------------------------------------------------------------------ filters
def filt(cases, dmax_km, lcs):
    cases = [int(c[5]) for c in cases]
    s = STOPS[STOPS.case.isin(cases) & (STOPS.distance_to_station_m <= dmax_km * 1000) & STOPS.land_cover.isin(lcs)]
    return s, PK[PK.stop_key.isin(s.stop_key)], LOST[LOST.case.isin(cases) & (LOST.dist_m <= dmax_km * 1000)]


# ------------------------------------------------------------------ overview
def kpis(s, p, l):
    def card(lab, val, sub, col):
        return (f"<div class='kpi' style='--accent:{col}'><div class='kl'>{lab}</div>"
                f"<div class='kv'>{val}</div><div class='ks'>{sub}</div></div>")
    n = len(s)
    dl = f"{100 * p.ok.mean():.1f}%" if len(p) else "–"
    ls = f"{100 * n / (n + len(l)):.0f}%" if n + len(l) else "–"
    sv = f"{s.saving_pct.median():.0f}%" if s.saving_pct.notna().any() else "–"
    return ("<div class='kpis'>" + card("Stops in view", n, f"of 61 · {len(p):,} packet slots", "#3B8FD9")
            + card("Delivered", dl, f"{int(p.ok.sum()):,} packets received", "#21B38A")
            + card("Link success", ls, f"{len(l)} lost locations in view", "#F0A73A")
            + card("Median energy saving", sv, "cheapest working setting vs SF12/14 dBm", "#E4572E")
            + card("Best transferable model", "0.67", "balanced accuracy on an unseen case", "#9B59B6") + "</div>")


def case_table(s, p, l):
    rows = []
    for c in sorted(s.case.unique()):
        sc, pc, lc = s[s.case == c], p[p.case == c], l[l.case == c]
        rows.append({"Case": CASE_NAME[c], "Stops": len(sc),
                     "Distance (m)": f"{sc.distance_to_station_m.min():.0f}–{sc.distance_to_station_m.max():.0f}",
                     "Deliv. %": round(100 * pc.ok.mean(), 1),
                     "PL₁₄ (dB)": round(sc.pl_median_14dbm.median(), 1),
                     "Lost": len(lc), "Link success %": round(100 * len(sc) / (len(sc) + len(lc)))})
    return pd.DataFrame(rows)


def link_plot(s, l):
    bands = [(0, 300, "< 0.3 km"), (300, 1000, "0.3–1 km"), (1000, 2000, "1–2 km"), (2000, 4000, "2–4 km")]
    ok_d, lost_d = s.distance_to_station_m.values, l.dist_m.values
    sw = [((ok_d >= a) & (ok_d < b)).sum() for a, b, _ in bands]
    lo = [((lost_d >= a) & (lost_d < b)).sum() for a, b, _ in bands]
    fig = go.Figure()
    fig.add_bar(x=[t for *_, t in bands], y=sw, name="sweep completed", marker_color="#21B38A")
    fig.add_bar(x=[t for *_, t in bands], y=lo, name="no link (lost)", marker_color="#E4572E")
    pct = [f"{100 * a / (a + b):.0f}%" if a + b else "" for a, b in zip(sw, lo)]
    fig.add_scatter(x=[t for *_, t in bands], y=[a + b + 1.2 for a, b in zip(sw, lo)], text=pct, mode="text",
                    textfont=dict(size=14, color=FONT), showlegend=False, hoverinfo="skip")
    fig.update_layout(barmode="stack", yaxis_title="Locations")
    return style(fig, "Link success falls with distance once lost locations are counted", 380)


# ------------------------------------------------------------------ delivery
def heatmap(p, split):
    if not len(p): return empty()
    groups = [(None, "All stops in view")] if split == "Pooled" else [(c, CASE_NAME[c]) for c in sorted(p.case.unique())]
    fig = make_subplots(1, len(groups), subplot_titles=[g[1] for g in groups], horizontal_spacing=0.06)
    for i, (c, _) in enumerate(groups, 1):
        d = p if c is None else p[p.case == c]
        z = d.pivot_table(index="sf", columns="pwr", values="ok", aggfunc="mean").reindex(index=SFS, columns=PWRS) * 100
        fig.add_trace(go.Heatmap(z=z.values, x=[f"{q} dBm" for q in PWRS], y=[f"SF{q}" for q in SFS], zmin=0, zmax=100,
                                 colorscale="Viridis", showscale=i == len(groups), text=np.round(z.values).astype("float"),
                                 texttemplate="%{text:.0f}%", colorbar=dict(title="%"),
                                 hovertemplate="%{y} · %{x}<br>delivered %{z:.1f}%<extra></extra>"), 1, i)
    return style(fig, "Packets delivered for each spreading factor and power", 420, False)


def sf_curve(p):
    if not len(p): return empty()
    fig = go.Figure()
    for q in PWRS:
        d = p[p.pwr == q].groupby("sf").ok.mean() * 100
        fig.add_scatter(x=d.index, y=d.values, mode="lines+markers", name=f"{q} dBm",
                        line=dict(color=PWR_COL[q], width=3), marker=dict(size=9),
                        hovertemplate=f"{q} dBm · SF%{{x}}<br>%{{y:.1f}}% delivered<extra></extra>")
    fig.update_layout(xaxis=dict(title="Spreading factor", dtick=1), yaxis=dict(title="Delivered (%)", range=[0, 100]))
    return style(fig, "Each SF step and power step buys delivery", 380)


def stop_card(stop):
    if stop is None or stop not in set(STOPS.stop_key):
        return "<div class='note'>Pick a stop.</div>", empty("Pick a stop")
    s = STOPS[STOPS.stop_key == stop].iloc[0]
    d = grid[grid.stop_key == stop].pivot(index="sf", columns="pwr", values="ok").reindex(index=SFS, columns=PWRS)
    fig = go.Figure(go.Heatmap(z=d.values, x=[f"{q} dBm" for q in PWRS], y=[f"SF{q}" for q in SFS], zmin=0, zmax=5,
                               colorscale=[[0, "#E4572E"], [.5, "#F0C93A"], [1, "#21B38A"]], text=d.values,
                               texttemplate="%{text}/5", colorbar=dict(title="of 5"),
                               hovertemplate="%{y} · %{x}: %{z}/5<extra></extra>"))
    if pd.notna(s.best_sf):
        fig.add_shape(type="rect", x0=PWRS.index(s.best_pwr) - .5, x1=PWRS.index(s.best_pwr) + .5,
                      y0=SFS.index(s.best_sf) - .5, y1=SFS.index(s.best_sf) + .5, line=dict(color="#ffffff", width=3))
    style(fig, f"Stop {stop}: received packets per setting", 380, False)
    best_txt = (f"SF{int(s.best_sf)} · {int(s.best_pwr)} dBm (saves {s.saving_pct:.0f}%)" if pd.notna(s.best_sf)
                else "none reached 4 of 5")
    pl = f"{s.pl_median_14dbm:.1f} dB" if pd.notna(s.pl_median_14dbm) else "no 14 dBm packet"
    html = (f"<div class='stopcard'><div class='sc-h' style='border-color:{CASE_COL[s.case]}'>{stop} · {CASE_NAME[s.case]}</div>"
            f"<div class='sc-g'><span>Distance</span><b>{s.distance_to_station_m:.0f} m</b>"
            f"<span>Elevation diff.</span><b>{s.elev_diff_m:+.0f} m</b>"
            f"<span>Land cover</span><b>{s.land_cover}</b>"
            f"<span>Delivered</span><b>{int(s.n_ok)}/90 ({100 * s.delivery_rate:.0f}%)</b>"
            f"<span>Median PL₁₄</span><b>{pl}</b>"
            f"<span>Cheapest working</span><b>{best_txt}</b></div></div>")
    return html, fig


# ------------------------------------------------------------------ path loss
def pl_plot(s, xvar, colour, fits, stop):
    d = s.dropna(subset=["pl_median_14dbm"])
    if not len(d): return empty(), ""
    x = "distance_to_station_m" if xvar == "Distance" else "elev_diff_m"
    fig = go.Figure()
    keys = sorted(d.case.unique()) if colour == "Case" else [k for k in LCS if k in set(d.land_cover)]
    for k in keys:
        dd = d[d.case == k] if colour == "Case" else d[d.land_cover == k]
        col = CASE_COL[k] if colour == "Case" else LC_COL[k]
        fig.add_scatter(x=dd[x], y=dd.pl_median_14dbm, mode="markers", name=CASE_NAME[k] if colour == "Case" else k,
                        marker=dict(size=11, color=col, line=dict(width=1, color="rgba(255,255,255,.6)")),
                        customdata=np.stack([dd.stop_key, dd.distance_to_station_m, dd.elev_diff_m, dd.land_cover,
                                             100 * dd.delivery_rate], -1),
                        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]:.0f} m · Δh %{customdata[2]:+.0f} m"
                                      "<br>%{customdata[3]} · %{customdata[4]:.0f}% delivered<br>PL₁₄ %{y:.1f} dB<extra></extra>")
    txt = []
    if fits and xvar == "Distance":
        groups = [("All in view", d, "#cbd5e1", "solid")] + [(CASE_NAME[c], d[d.case == c], CASE_COL[c], "dash")
                                                              for c in sorted(d.case.unique()) if (d.case == c).sum() >= 5]
        for lab, dd, col, dash in groups:
            b = np.polyfit(10 * dd.logd, dd.pl_median_14dbm, 1)
            xs = np.logspace(np.log10(dd[x].min()), np.log10(dd[x].max()), 40)
            fig.add_scatter(x=xs, y=b[1] + b[0] * 10 * np.log10(xs), mode="lines", line=dict(color=col, dash=dash, width=2.5),
                            name=f"{lab}: n = {b[0]:.2f}", hoverinfo="skip")
            txt.append(f"**{lab}** n = {b[0]:.2f}")
    if stop in set(d.stop_key):
        r = d[d.stop_key == stop].iloc[0]
        fig.add_scatter(x=[r[x]], y=[r.pl_median_14dbm], mode="markers", name=f"selected {stop}",
                        marker=dict(size=22, color="rgba(0,0,0,0)", line=dict(width=3, color="#E4572E")), hoverinfo="skip")
    fig.add_hline(y=148, line_dash="dot", line_color=FONT, annotation_text="measurable limit (SF12, 14 dBm)",
                  annotation_font_color=FONT)
    fig.update_xaxes(type="log" if xvar == "Distance" else "linear",
                     title="Distance to Station (m, log)" if xvar == "Distance" else "Stop elevation − Station elevation (m)")
    fig.update_yaxes(title="Median path loss at 14 dBm (dB)")
    style(fig, "Path loss by stop (hover for details; ringed = selected stop)", 500)
    return fig, (" · ".join(txt) + "  \nPooled n ≈ 0 means distance alone explains almost nothing here." if txt else "")


# ------------------------------------------------------------------ settings advisor
def _mk_tree(cols):
    return DecisionTreeClassifier(max_depth=3, min_samples_leaf=90, random_state=0).fit(PK[cols], PK.ok)


ADV_SFP = _mk_tree(FEATURE_SETS["SF + power"])
ADV_LR = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(PK[ALL], PK.ok)


def advisor(dist, dh, lc, target, model):
    rows = []
    for s_ in SFS:
        for p_ in PWRS:
            x = {"sf": s_, "pwr": p_, "logd": np.log10(max(dist, 20)), "elev_diff_m": dh,
                 "elev_angle_deg": np.degrees(np.arctan((dh + 1.3 - 2.0) / max(dist, 20))),
                 "lc_built-up": int(lc == "built-up"), "lc_mixed": int(lc == "mixed"), "lc_open": int(lc == "open")}
            X = pd.DataFrame([x])
            pr = (ADV_SFP.predict_proba(X[FEATURE_SETS["SF + power"]]) if model.startswith("Transferable")
                  else ADV_LR.predict_proba(X[ALL]))[0, 1]
            rows.append((s_, p_, pr))
    r = pd.DataFrame(rows, columns=["sf", "pwr", "p"])
    z = r.pivot(index="sf", columns="pwr", values="p").reindex(index=SFS, columns=PWRS)
    fig = go.Figure(go.Heatmap(z=z.values * 100, x=[f"{q} dBm" for q in PWRS], y=[f"SF{q}" for q in SFS], zmin=0, zmax=100,
                               colorscale="Viridis", text=z.values * 100, texttemplate="%{text:.0f}%",
                               colorbar=dict(title="%"), hovertemplate="%{y} · %{x}: %{z:.0f}%<extra></extra>"))
    ok = r[r.p >= target].copy()
    if len(ok):
        ok["cost"] = COST(ok.sf, ok.pwr); b = ok.sort_values("cost").iloc[0]
        save = 100 * (1 - b.cost / FULL_COST)
        save_txt = (f"{save:.0f}% less energy than SF12/14 dBm" if save > 0.5
                    else "only the strongest setting reaches this target here; lower the target to see cheaper options")
        fig.add_shape(type="rect", x0=PWRS.index(b.pwr) - .5, x1=PWRS.index(b.pwr) + .5,
                      y0=SFS.index(b.sf) - .5, y1=SFS.index(b.sf) + .5, line=dict(color="#ffffff", width=4))
        rec = (f"<div class='rec ok'><div class='rk'>Recommended setting</div><div class='rv'>SF{int(b.sf)} · {int(b.pwr)} dBm</div>"
               f"<div class='rs'>predicted delivery {100 * b.p:.0f}% · {save_txt}"
               f" · airtime ×{2 ** (b.sf - 7):.0f} vs SF7</div></div>")
    else:
        rec = ("<div class='rec bad'><div class='rk'>No setting reaches the target</div><div class='rv'>Use SF12 · 14 dBm</div>"
               "<div class='rs'>and test on site: the model predicts this place is marginal or a dead zone</div></div>")
    style(fig, f"Predicted delivery at {dist:.0f} m, Δh {dh:+.0f} m, {lc}", 420, False)
    nb = STOPS.assign(score=np.abs(np.log10(STOPS.distance_to_station_m / max(dist, 20))) * 3 + np.abs(STOPS.elev_diff_m - dh) / 50)
    nb = nb.nsmallest(5, "score")
    tab = pd.DataFrame({"Stop": nb.stop_key, "Case": nb.case, "Distance (m)": nb.distance_to_station_m.round(0),
                        "Δh (m)": nb.elev_diff_m, "Land cover": nb.land_cover,
                        "Delivered %": (100 * nb.delivery_rate).round(0),
                        "Measured cheapest": [f"SF{int(a)} · {int(b_)} dBm" if pd.notna(a) else "none"
                                              for a, b_ in zip(nb.best_sf, nb.best_pwr)]})
    return rec, fig, tab


# ------------------------------------------------------------------ model lab
MODELS = {
    "Decision tree": lambda depth: DecisionTreeClassifier(max_depth=depth, min_samples_leaf=90, random_state=0),
    "Logistic regression": lambda depth: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "Random forest (100 trees)": lambda depth: RandomForestClassifier(100, random_state=0, n_jobs=-1),
    "Gradient boosting": lambda depth: HistGradientBoostingClassifier(random_state=0),
}
SPLITS = {"Grouped by stop (10-fold)": "stop", "Leave one case out": "case", "Random packets (leaky!)": "random"}


@lru_cache(maxsize=64)
def cv_probs(fs, model, depth, split):
    cols, y = FEATURE_SETS[fs], PK.ok.values
    m = MODELS[model](depth)
    t0 = time.perf_counter()
    if split == "stop":
        pr = cross_val_predict(m, PK[cols], y, groups=PK.stop_key, cv=GroupKFold(10), method="predict_proba")[:, 1]
    elif split == "case":
        pr = cross_val_predict(m, PK[cols], y, groups=PK.case, cv=LeaveOneGroupOut(), method="predict_proba")[:, 1]
    else:
        pr = cross_val_predict(m, PK[cols], y, cv=KFold(10, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    return pr, time.perf_counter() - t0


def lab(fs, model, depth, split_lab, thr):
    split = SPLITS[split_lab]
    y = PK.ok.values
    pr, secs = cv_probs(fs, model, int(depth), split)
    yh = pr >= thr
    ba, auc = balanced_accuracy_score(y, yh), roc_auc_score(y, pr)
    pre, rec, f1 = precision_score(y, yh, zero_division=0), recall_score(y, yh), f1_score(y, yh)
    def m(lab_, v, good):
        return f"<div class='metric {'g' if good else ''}'><div class='ml'>{lab_}</div><div class='mv'>{v:.2f}</div></div>"
    html = ("<div class='metrics'>" + m("Balanced accuracy", ba, ba >= .66) + m("AUC", auc, auc >= .7) + m("Precision", pre, pre >= .68)
            + m("Recall", rec, rec >= .6) + m("F1", f1, f1 >= .63) + "</div>"
            + f"<div class='note'>{split_lab} · threshold {thr:.2f} · computed in {secs:.1f} s (cached after the first run)."
            + (" <b>Random split puts packets of the same stop in train and test, so this score is inflated.</b>" if split == "random" else "")
            + "</div>")
    fpr, tpr, th = roc_curve(y, pr)
    i = np.argmin(np.abs(th - thr))
    roc = go.Figure()
    roc.add_scatter(x=fpr, y=tpr, mode="lines", line=dict(color="#3B8FD9", width=3), name=f"ROC (AUC {auc:.2f})",
                    hovertemplate="FPR %{x:.2f} · TPR %{y:.2f}<extra></extra>")
    roc.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=FONT, dash="dot"), name="chance", hoverinfo="skip")
    roc.add_scatter(x=[fpr[i]], y=[tpr[i]], mode="markers", marker=dict(size=14, color="#E4572E"), name=f"threshold {thr:.2f}")
    roc.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate")
    style(roc, "ROC curve", 380)
    cm = confusion_matrix(y, yh)
    cmf = go.Figure(go.Heatmap(z=cm, x=["predicted lost", "predicted delivered"], y=["lost", "delivered"], colorscale="Blues",
                               text=cm, texttemplate="%{text:,}", showscale=False, hovertemplate="%{y} → %{x}: %{z:,}<extra></extra>"))
    style(cmf, "Confusion matrix", 380, False)
    rules = ""
    if model == "Decision tree":
        t = DecisionTreeClassifier(max_depth=int(depth), min_samples_leaf=90, random_state=0).fit(PK[FEATURE_SETS[fs]], y)
        rules = "```text\n" + export_text(t, feature_names=FEATURE_SETS[fs], decimals=1) + "```"
    return html, roc, cmf, rules


# ------------------------------------------------------------------ dead-zone map (synthetic terrain)
N, SIZE = 161, 4000.0
XS = np.linspace(0, SIZE, N)
GX, GY = np.meshgrid(XS, XS)


@lru_cache(maxsize=16)
def terrain(seed, rugged):
    rng = np.random.default_rng(int(seed))
    vx = 2000 + 350 * np.sin(GY / 900)
    z = (1250 + 0.00004 * (GX - vx) ** 2
         + 260 * rugged * np.exp(-((GX - 500) / 420) ** 2) * (0.7 + 0.3 * np.sin(GY / 650))
         + 300 * rugged * np.exp(-((GX - 3550) / 450) ** 2) * (0.75 + 0.25 * np.cos(GY / 800))
         + 180 * rugged * np.exp(-((GX - 1350) / 220) ** 2 - ((GY - 2300) / 500) ** 2)
         + 220 * rugged * np.exp(-((GX - 2900) / 350) ** 2 - ((GY - 800) / 350) ** 2)
         + gaussian_filter(rng.normal(0, 1, (N, N)), 5) * 60 * rugged)
    return z


def elev_at(Z, x, y):
    return float(map_coordinates(Z, [[y / SIZE * (N - 1)], [x / SIZE * (N - 1)]], order=1)[0])


def los(Z, sx, sy, s_ant, r_ant=1.3, samples=80):
    zs = elev_at(Z, sx, sy) + s_ant
    t = np.linspace(0.02, 0.98, samples)[:, None, None]
    px, py = sx + t * (GX - sx), sy + t * (GY - sy)
    terr = map_coordinates(Z, [py.ravel() / SIZE * (N - 1), px.ravel() / SIZE * (N - 1)], order=1).reshape(px.shape)
    return ~(terr > zs + t * ((Z + r_ant) - zs)).any(axis=0)


TREE_ALL = _mk_tree(ALL)


def cell_X(Z, sx, sy, s_ant, sf, pwr, lc):
    d = np.maximum(np.hypot(GX - sx, GY - sy), 20.0)
    ed = Z - elev_at(Z, sx, sy)
    return pd.DataFrame({"sf": sf, "pwr": pwr, "logd": np.log10(d).ravel(), "elev_diff_m": ed.ravel(),
                         "elev_angle_deg": np.degrees(np.arctan((ed + 1.3 - s_ant) / d)).ravel(),
                         "lc_built-up": int(lc == "built-up"), "lc_mixed": int(lc == "mixed"), "lc_open": int(lc == "open")})


def dead_zone(seed, rugged, sx, sy, s_ant, sf, pwr, lc, layer, thr):
    Z = terrain(int(seed), float(rugged))
    sx, sy = float(sx), float(sy)
    clear = los(Z, sx, sy, s_ant)
    if layer.startswith("Cheapest"):
        bestk = np.full(N * N, np.nan)
        for k, (_, s_, p_) in enumerate(COSTS):
            pr = TREE_ALL.predict_proba(cell_X(Z, sx, sy, s_ant, s_, p_, lc)[ALL])[:, 1]
            bestk[np.isnan(bestk) & (pr >= thr)] = k
        val = bestk.reshape(N, N); dead = np.isnan(val)
        cs, cmin, cmax, ctitle = "Viridis", 0, len(COSTS) - 1, "cost rank"
        sub = f"cheapest setting reaching {thr:.0%}"
    else:
        val = TREE_ALL.predict_proba(cell_X(Z, sx, sy, s_ant, sf, pwr, lc)[ALL])[:, 1].reshape(N, N)
        raw_ml = val.copy()
        if layer.startswith("ML + line"):
            val = np.where(clear, val, val * 0.5)
        dead = val < thr
        cs, cmin, cmax, ctitle = [[0, "#B2182B"], [.5, "#F7F7F7"], [1, "#2166AC"]], 0, 1, "P(delivery)"
        sub = f"SF{sf} · {pwr} dBm"
    if layer.startswith("Line of sight"):
        val = clear.astype(float); dead = ~clear
        cs, cmin, cmax, ctitle = [[0, "#B2182B"], [1, "#2166AC"]], 0, 1, "clear LoS"
        sub = "geometric line of sight only"
    zs = elev_at(Z, sx, sy)
    surf = np.where(np.isnan(val), cmin - 1, val)
    f3 = go.Figure(go.Surface(x=XS, y=XS, z=Z, surfacecolor=surf, colorscale=cs, cmin=cmin, cmax=cmax,
                              colorbar=dict(title=ctitle, len=.6), lighting=dict(ambient=.55, diffuse=.8, roughness=.9, specular=.1),
                              contours=dict(z=dict(show=True, start=1200, end=2000, size=50, color="rgba(40,40,40,.35)")),
                              hovertemplate="x %{x:.0f} · y %{y:.0f}<br>elev %{z:.0f} m<extra></extra>"))
    k = 8; dy, dx = np.where(dead[::k, ::k])
    f3.add_scatter3d(x=XS[::k][dx], y=XS[::k][dy], z=Z[::k, ::k][dy, dx] + 6, mode="markers",
                     marker=dict(size=2.2, color="rgba(120,0,0,.85)"), name="dead point", hoverinfo="skip")
    mast = 60 + 8 * s_ant
    f3.add_scatter3d(x=[sx, sx], y=[sy, sy], z=[zs, zs + mast], mode="lines", line=dict(color="#111", width=7), name="mast", hoverinfo="skip")
    f3.add_scatter3d(x=[sx], y=[sy], z=[zs + mast], mode="markers+text", text=["Station"], textposition="top center",
                     marker=dict(size=7, color="gold", symbol="diamond"), name="Station", textfont=dict(color=FONT))
    f3.update_layout(scene=dict(aspectmode="manual", aspectratio=dict(x=1.25, y=1.25, z=.42),
                                xaxis=dict(title="x (m)", nticks=5, color=FONT), yaxis=dict(title="y (m)", nticks=5, color=FONT),
                                zaxis=dict(title="elev (m)", nticks=4, color=FONT),
                                camera=dict(eye=dict(x=1.05, y=-1.05, z=.8), projection=dict(type="orthographic"))))
    style(f3, f"Isometric view · {sub}", 620)
    f2 = go.Figure(go.Heatmap(x=XS, y=XS, z=val, colorscale=cs, zmin=cmin, zmax=cmax, opacity=.8, colorbar=dict(title=ctitle),
                              hovertemplate="x %{x:.0f} · y %{y:.0f}<extra></extra>"))
    f2.add_contour(x=XS, y=XS, z=Z, showscale=False, contours=dict(coloring="none", showlabels=True, start=1200, end=2000, size=50,
                   labelfont=dict(size=9, color="#444")), line=dict(color="rgba(60,60,60,.45)", width=1), hoverinfo="skip",
                   name="contours (m)", showlegend=True)
    dy, dx = np.where(dead[::6, ::6])
    f2.add_scatter(x=XS[::6][dx], y=XS[::6][dy], mode="markers", name="dead point",
                   marker=dict(symbol="x", size=4, color="rgba(120,0,0,.6)"), hoverinfo="skip")
    f2.add_scatter(x=[sx], y=[sy], mode="markers", name="Station", marker=dict(symbol="star", size=20, color="gold",
                   line=dict(color="#111", width=1)))
    f2.update_layout(xaxis=dict(title="x (m)", scaleanchor="y", constrain="domain"), yaxis=dict(title="y (m)"))
    style(f2, f"Top view · {sub}", 620)
    cover = 100 * (1 - dead.mean())
    disagree = 100 * ((raw_ml >= thr) & ~clear).mean() if not layer.startswith(("Cheapest", "Line")) else None
    html = ("<div class='metrics'>"
            f"<div class='metric g'><div class='ml'>Coverage</div><div class='mv'>{cover:.0f}%</div></div>"
            f"<div class='metric'><div class='ml'>Dead points</div><div class='mv'>{100 - cover:.0f}%</div></div>"
            f"<div class='metric'><div class='ml'>Clear line of sight</div><div class='mv'>{100 * clear.mean():.0f}%</div></div>"
            f"<div class='metric'><div class='ml'>Station ground</div><div class='mv'>{zs:.0f} m</div></div>"
            + (f"<div class='metric'><div class='ml'>ML says OK, LoS blocked</div><div class='mv'>{disagree:.0f}%</div></div>" if disagree is not None else "")
            + "</div><div class='note'>The tree sees distance, elevation difference and angle, not the terrain <i>between</i> the antennas. "
              "Where the ML layer and the line-of-sight layer disagree is exactly the gap the paper identifies. Synthetic terrain, not a real place.</div>")
    return f3, f2, html


# ------------------------------------------------------------------ results tables
REG = pd.DataFrame([["Mean only", 11.1, 11.4], ["Log-distance", 11.3, 13.0], ["+ elevation (OLS)", 11.4, 15.8],
                    ["+ land cover (ridge)", 11.1, 15.7], ["Regression tree", 11.2, 15.3], ["Random forest", 11.3, 15.0]],
                   columns=["Model", "LOO RMSE (dB)", "LOCO RMSE (dB)"])
CLF = pd.DataFrame([["Link-budget rule", .54, .73, None], ["Tree, SF + power", .68, .71, .67], ["Tree, all features", .62, .64, .50],
                    ["Random forest, all", .65, .71, .55], ["Logistic, SF + power", .68, .73, .65], ["Gradient boosting, all", .68, .75, .58]],
                   columns=["Model", "Balanced acc. (grouped)", "AUC", "Balanced acc. (unseen case)"])


def results_fig():
    fig = make_subplots(1, 2, subplot_titles=["Path loss RMSE (lower is better)", "Delivery balanced accuracy"], horizontal_spacing=.18)
    fig.add_bar(y=REG.Model, x=REG["LOO RMSE (dB)"], orientation="h", name="leave one stop out", marker_color="#3B8FD9", row=1, col=1)
    fig.add_bar(y=REG.Model, x=REG["LOCO RMSE (dB)"], orientation="h", name="unseen case", marker_color="#F0A73A", row=1, col=1)
    c = CLF.dropna()
    fig.add_bar(y=c.Model, x=c["Balanced acc. (grouped)"], orientation="h", name="grouped by stop", marker_color="#21B38A",
                row=1, col=2)
    fig.add_bar(y=c.Model, x=c["Balanced acc. (unseen case)"], orientation="h", name="unseen case ", marker_color="#E4572E",
                row=1, col=2)
    fig.add_vline(x=.5, line_dash="dot", line_color=FONT, row=1, col=2)
    fig.update_xaxes(range=[.45, .75], row=1, col=2)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(barmode="group")
    return style(fig, "Radio settings transfer; terrain features and extra model capacity do not", 460)


def export_csv():
    cols = ["stop_key", "case", "distance_to_station_m", "elev_diff_m", "elev_angle_deg", "land_cover", "n_ok", "delivery_rate",
            "pl_median_14dbm", "best_sf", "best_pwr", "saving_pct"]
    path = os.path.join(tempfile.gettempdir(), "lora_stops_no_coordinates.csv")
    STOPS[cols].sort_values("stop_key", key=lambda s: s.map(stop_sort)).to_csv(path, index=False)
    return path


# ------------------------------------------------------------------ UI
CSS = """
.gradio-container{max-width:1380px!important}
#hero{padding:18px 22px;border-radius:16px;background:linear-gradient(120deg,#0f2e4d 0%,#1d4f7a 55%,#21806b 100%);color:#fff;margin-bottom:6px}
#hero h1{margin:0;font-size:28px;color:#fff}#hero p{margin:6px 0 0;opacity:.88;color:#fff}
#hero .chips{margin-top:10px;display:flex;gap:8px;flex-wrap:wrap}
#hero .chip{background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.25);border-radius:999px;padding:3px 11px;font-size:12px;color:#fff}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}
.kpi{border:1px solid var(--border-color-primary);border-left:5px solid var(--accent);border-radius:12px;padding:12px 14px;background:var(--block-background-fill)}
.kl{font-size:12px;opacity:.75}.kv{font-size:28px;font-weight:800;margin-top:2px;color:var(--accent)}.ks{font-size:11.5px;opacity:.7}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin:4px 0}
.metric{border:1px solid var(--border-color-primary);border-radius:10px;padding:10px 12px;background:var(--block-background-fill)}
.metric .ml{font-size:12px;opacity:.75}.metric .mv{font-size:24px;font-weight:800}.metric.g .mv{color:#21B38A}
.note{font-size:13px;opacity:.85;margin:6px 2px}
.rec{border-radius:14px;padding:16px 18px;color:#fff}.rec.ok{background:linear-gradient(120deg,#157a5f,#21B38A)}
.rec.bad{background:linear-gradient(120deg,#9f2a14,#E4572E)}.rk{font-size:12px;opacity:.9;text-transform:uppercase;letter-spacing:.6px}
.rv{font-size:30px;font-weight:800;margin:2px 0}.rs{font-size:13px;opacity:.95}
.stopcard{border:1px solid var(--border-color-primary);border-radius:12px;padding:12px 14px;background:var(--block-background-fill)}
.sc-h{font-weight:800;font-size:16px;border-left:5px solid;padding-left:8px;margin-bottom:8px}
.sc-g{display:grid;grid-template-columns:auto 1fr;gap:5px 14px;font-size:13.5px}.sc-g span{opacity:.7}
.filters{border:1px solid var(--border-color-primary);border-radius:14px;padding:6px 10px}
"""

HERO = """<div id='hero'><h1>📡 LoRa Hill-Terrain Explorer</h1>
<p>61 field sweeps at 866 MHz in three hill-terrain case studies near Kathmandu: what gets packets through, what predicts it, and where the dead zones are.</p>
<div class='chips'><span class='chip'>18 radio settings per stop</span><span class='chip'>5,490 packet slots</span>
<span class='chip'>grouped + leave-one-case-out validation</span><span class='chip'>no coordinates shown</span></div></div>"""

METHOD = """
#### How the data were collected
A fixed **Station** sends 6 spreading factors × 3 powers × 5 packets (90 slots, ≈67 s) at each stop; the walking **Roamer** logs every slot. A failed handshake means no sweep: a *lost location*.

#### How models were tested
Packets from one stop are never split between training and test (**grouped by stop**), and **leave-one-case-out** asks whether a model works on a terrain it never saw. A random packet split is shown only to demonstrate leakage.

#### What this dashboard is not
- Not a coverage-planning tool. The dead-zone map applies a tree trained on 61 stops to **invented** terrain.
- The tree never sees the terrain between the antennas; the line-of-sight layer is a separate geometric check.
- Results describe three cases near one area in one season.
"""

STOP_CHOICES = sorted(STOPS.stop_key, key=stop_sort)
GR6 = int(gr.__version__.split(".")[0]) >= 6
THEME = gr.themes.Soft(primary_hue="blue", secondary_hue="emerald", neutral_hue="slate",
                       font=[gr.themes.Font("Inter"), gr.themes.Font("Segoe UI"), gr.themes.Font("system-ui"), gr.themes.Font("sans-serif")])

with gr.Blocks(title="LoRa Hill-Terrain Explorer", **({} if GR6 else {"css": CSS, "theme": THEME})) as demo:
    gr.HTML(HERO)
    with gr.Group(elem_classes="filters"):
        with gr.Row():
            f_case = gr.CheckboxGroup([CASE_NAME[c] for c in (1, 2, 3)], value=[CASE_NAME[c] for c in (1, 2, 3)],
                                      label="Cases", scale=3)
            f_dist = gr.Slider(0.1, 3.5, 3.5, step=0.1, label="Max distance to Station (km)", scale=2)
            f_lc = gr.CheckboxGroup(LCS, value=LCS, label="Land cover at stop", scale=3)
    kpi_html = gr.HTML()

    with gr.Tabs():
        with gr.Tab("Overview"):
            with gr.Row():
                with gr.Column(scale=3):
                    case_df = gr.Dataframe(wrap=True, interactive=False, label="Case profiles (filtered)",
                                           column_widths=["28%", "9%", "15%", "12%", "12%", "9%", "15%"])
                    link_fig = gr.Plot()
                with gr.Column(scale=2):
                    gr.Markdown("### Key findings\n"
                                "1. **Settings transfer.** A depth-3 tree on SF and power scores 0.67 on an unseen terrain.\n"
                                "2. **Distance alone fails.** No path loss model beats the mean (RMSE 11.1 dB).\n"
                                "3. **Hidden failures.** Counting lost locations, link success falls from 91% to 52%.\n"
                                "4. **Energy.** The cheapest working setting saves a median 97% vs SF12 at 14 dBm.\n"
                                "5. **Simple wins.** The 2 kB tree beats a 24 MB forest on new terrain.\n\n"
                                "> In hill terrain, what lies between the antennas matters more than the distance between them.")
                    gr.Markdown(f"<div class='note'>Data source: {DATA_SOURCE}</div>")

        with gr.Tab("Delivery explorer"):
            split = gr.Radio(["Pooled", "By case"], value="By case", label="Heatmap layout")
            heat_fig = gr.Plot()
            with gr.Row():
                curve_fig = gr.Plot()
                with gr.Column():
                    stop_dd = gr.Dropdown(STOP_CHOICES, value="B10", label="Inspect one stop (also ringed on the Path loss tab)")
                    stop_html = gr.HTML()
                    stop_fig = gr.Plot()

        with gr.Tab("Path loss"):
            with gr.Row():
                xvar = gr.Radio(["Distance", "Elevation difference"], value="Distance", label="x-axis")
                colour = gr.Radio(["Case", "Land cover"], value="Case", label="Colour by")
                fits = gr.Checkbox(True, label="Show log-distance fits")
            pl_fig = gr.Plot()
            pl_md = gr.Markdown()

        with gr.Tab("Settings advisor"):
            gr.Markdown("Describe a new node location and get the cheapest setting predicted to reach your target. "
                        "The five most similar measured stops are shown as evidence.")
            with gr.Row():
                with gr.Column(scale=1):
                    a_dist = gr.Slider(20, 3500, 1200, step=10, label="Distance to gateway (m)")
                    a_dh = gr.Slider(-50, 300, 30, step=5, label="Node elevation − gateway elevation (m)")
                    a_lc = gr.Dropdown(LCS, value="mixed", label="Land cover at node")
                    a_t = gr.Slider(0.5, 0.95, 0.7, step=0.05, label="Target delivery probability")
                    a_m = gr.Radio(["Transferable tree (SF + power)", "Logistic regression (all features)"],
                                   value="Logistic regression (all features)", label="Model")
                with gr.Column(scale=2):
                    a_rec = gr.HTML()
                    a_fig = gr.Plot()
            a_tab = gr.Dataframe(wrap=True, interactive=False, label="Most similar measured stops")

        with gr.Tab("Model lab"):
            with gr.Row():
                m_fs = gr.Dropdown(list(FEATURE_SETS), value="SF + power", label="Features")
                m_m = gr.Dropdown(list(MODELS), value="Decision tree", label="Model")
                m_split = gr.Radio(list(SPLITS), value="Grouped by stop (10-fold)", label="Validation")
            with gr.Row():
                m_depth = gr.Slider(1, 8, 3, step=1, label="Tree depth (decision tree only)")
                m_thr = gr.Slider(0.1, 0.9, 0.5, step=0.05, label="Decision threshold")
            m_html = gr.HTML()
            with gr.Row():
                m_roc = gr.Plot()
                m_cm = gr.Plot()
            with gr.Accordion("Learned tree rules", open=False):
                m_rules = gr.Markdown()

        with gr.Tab("Dead-zone map"):
            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    d_layer = gr.Radio(["ML prediction (tree)", "ML + line-of-sight check", "Line of sight only",
                                        "Cheapest working setting"], value="ML + line-of-sight check", label="Layer")
                    d_sx = gr.Slider(0, 4000, 1900, step=25, label="Station x (m)")
                    d_sy = gr.Slider(0, 4000, 1600, step=25, label="Station y (m)")
                    d_ant = gr.Slider(1, 15, 2, step=0.5, label="Station antenna height (m)")
                    with gr.Row():
                        d_sf = gr.Dropdown(SFS, value=9, label="SF")
                        d_pw = gr.Dropdown(PWRS, value=8, label="Power (dBm)")
                    d_lc = gr.Dropdown(LCS, value="mixed", label="Land cover at receivers")
                    d_thr = gr.Slider(0.3, 0.85, 0.5, step=0.05, label="Working-link threshold")
                    with gr.Accordion("Terrain", open=False):
                        d_seed = gr.Slider(1, 50, 7, step=1, label="Terrain seed")
                        d_rug = gr.Slider(0.3, 1.8, 1.0, step=0.1, label="Ruggedness")
                with gr.Column(scale=3):
                    d_html = gr.HTML()
                    with gr.Tabs():
                        with gr.Tab("Isometric 3-D"):
                            d_3d = gr.Plot()
                        with gr.Tab("Top view"):
                            d_2d = gr.Plot()

        with gr.Tab("Results & method"):
            gr.Plot(results_fig())
            with gr.Row():
                gr.Dataframe(REG, wrap=True, interactive=False, label="Path loss regression (60 stops)")
                gr.Dataframe(CLF, wrap=True, interactive=False, label="Delivery classifiers (5,490 packets)")
            gr.Markdown(METHOD)
            dl = gr.DownloadButton("Download stop table (CSV, no coordinates)", value=export_csv)

    # ---------------- wiring
    F = [f_case, f_dist, f_lc]

    def refresh(cases, dmax, lcs, sp, stop, xv, col, ft):
        s, p, l = filt(cases, dmax, lcs)
        choices = sorted(s.stop_key, key=stop_sort) or STOP_CHOICES
        stop = stop if stop in choices else choices[0]
        pf, pm = pl_plot(s, xv, col, ft, stop)
        return (kpis(s, p, l), case_table(s, p, l), link_plot(s, l), heatmap(p, sp), sf_curve(p),
                gr.update(choices=choices, value=stop), pf, pm)

    R_IN = F + [split, stop_dd, xvar, colour, fits]
    R_OUT = [kpi_html, case_df, link_fig, heat_fig, curve_fig, stop_dd, pl_fig, pl_md]
    for comp in F + [split, xvar, colour, fits]:
        comp.change(refresh, R_IN, R_OUT)

    def on_stop(stop, cases, dmax, lcs, xv, col, ft):
        s, _, _ = filt(cases, dmax, lcs)
        h, f = stop_card(stop)
        pf, pm = pl_plot(s, xv, col, ft, stop)
        return h, f, pf, pm
    stop_dd.change(on_stop, [stop_dd] + F + [xvar, colour, fits], [stop_html, stop_fig, pl_fig, pl_md])

    A_IN = [a_dist, a_dh, a_lc, a_t, a_m]
    for comp in A_IN:
        comp.change(advisor, A_IN, [a_rec, a_fig, a_tab])

    M_IN = [m_fs, m_m, m_depth, m_split, m_thr]
    for comp in M_IN:
        comp.change(lab, M_IN, [m_html, m_roc, m_cm, m_rules])

    D_IN = [d_seed, d_rug, d_sx, d_sy, d_ant, d_sf, d_pw, d_lc, d_layer, d_thr]
    for comp in D_IN:
        comp.release(dead_zone, D_IN, [d_3d, d_2d, d_html]) if isinstance(comp, gr.Slider) else comp.change(dead_zone, D_IN, [d_3d, d_2d, d_html])

    def init():
        return (refresh(*[c.value for c in R_IN])
                + stop_card("B10") + advisor(1200, 30, "mixed", .7, "Logistic regression (all features)")
                + lab("SF + power", "Decision tree", 3, "Grouped by stop (10-fold)", .5)
                + dead_zone(7, 1.0, 1900, 1600, 2, 9, 8, "mixed", "ML + line-of-sight check", .5))
    demo.load(init, None, R_OUT + [stop_html, stop_fig, a_rec, a_fig, a_tab, m_html, m_roc, m_cm, m_rules, d_3d, d_2d, d_html])

if __name__ == "__main__":
    kw = dict(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
    demo.launch(**({"css": CSS, "theme": THEME} if GR6 else {}), **kw)
