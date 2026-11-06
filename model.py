import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as H
from sklearn.metrics import roc_auc_score, average_precision_score


# --- settings ---
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


# --- load and clean ---
df = pd.read_csv(sys.argv[1], low_memory=False)
df["st"] = pd.to_datetime(df.Start_Time, format="mixed")
df["yr"] = df.st.dt.year
d = df.Description.fillna("")
df = df[
    ~d.str.contains(r"^\s*(?:construction|road ?work)", case=False, regex=True) & (df.yr <= 2022)
].copy()
s = df.Street.fillna("")
df["interstate"] = s.str.contains(r"\bI-\d|Interstate", regex=True)
df["night"] = df.Civil_Twilight.eq("Night")
df["wet"] = df.Weather_Condition.fillna("").str.contains(
    "Rain|Drizzle|Storm|Snow|Sleet|Hail|Ice|Wintry|Mix", case=False, regex=True
)
df["peak"] = df.st.dt.hour.isin([6, 7, 8, 15, 16, 17, 18])
df["la"] = np.round(df.Start_Lat / 0.01).astype(int)
df["lo"] = np.round(df.Start_Lng / 0.01).astype(int)
df["site"] = df.la.astype(str) + "_" + df.lo.astype(str)
tags = ["Traffic_Signal", "Stop", "Junction", "Crossing", "Station", "Amenity"]


# --- build site features ---
def build(end):
    p = df[df.yr <= end]
    X = p.groupby("site").agg(
        past=("ID", "size"),
        la=("la", "first"),
        lo=("lo", "first"),
        state=("State", "first"),
        n_years=("yr", "nunique"),
        recent=("yr", lambda y: (y >= end - 1).sum()),
        src1=("Source", lambda x: (x == "Source1").mean()),
        night=("night", "mean"),
        wet=("wet", "mean"),
        peak=("peak", "mean"),
        interstate=("interstate", "mean"),
        **{t: (t, "mean") for t in tags},
    )
    idx = {(a, b): c for a, b, c in zip(X.la, X.lo, X.past)}
    X["neighbors"] = [
        sum(idx.get((a + i, b + j), 0) for i in (-1, 0, 1) for j in (-1, 0, 1) if i or j)
        for a, b in zip(X.la, X.lo)
    ]
    st = p.groupby("State").yr
    X["state_trend"] = X.state.map((st.apply(lambda y: (y >= end - 1).sum()) / st.size()))
    X = X[X.past >= 5].copy()
    f = df[df.yr.between(end + 1, end + 2)].site.value_counts().reindex(X.index).fillna(0)
    return X, f


# --- train on older years, test on newer years ---
Xa, fa = build(2018)
Xb, fb = build(2020)
ya = (fa >= fa.quantile(0.95)).astype(int)
yb = (fb >= fb.quantile(0.95)).astype(int)
feats = [c for c in Xa.columns if c not in ("la", "lo", "state")]
print("train sites", len(Xa), "test sites", len(Xb), "test positives", int(yb.sum()))
p = np.mean(
    [
        H(max_depth=4, learning_rate=0.05, max_iter=200, random_state=r)
        .fit(Xa[feats], ya)
        .predict_proba(Xb[feats])[:, 1]
        for r in range(3)
    ],
    axis=0,
)
ml = pd.Series(p, index=Xb.index)
base = Xb.past.astype(float)


# --- compare random, past counts and the model ---
k = int(yb.sum())


def m(s):
    t = s.sort_values(ascending=False, kind="stable").index[:k]
    return dict(
        auc=round(roc_auc_score(yb, s), 3),
        ap=round(average_precision_score(yb, s), 3),
        precision=round(yb.loc[t].mean(), 3),
        captured=round(fb.loc[t].sum() / fb.sum(), 3),
    )


rng = np.random.default_rng(0)
R = {
    "Random": m(pd.Series(rng.random(len(yb)), index=yb.index)),
    "Past count": m(base + rng.random(len(base)) * 1e-6),
    "ML": m(ml),
}
print(pd.DataFrame(R).T)


# --- bootstrap the gain ---
v = []
a = ml.values
b = (base + rng.random(len(base)) * 1e-6).values
y = yb.values
n = len(y)
for _ in range(500):
    i = rng.integers(0, n, n)
    ai, bi, yi = a[i], b[i], y[i]
    v.append(yi[np.argsort(-ai)[:k]].mean() - yi[np.argsort(-bi)[:k]].mean())
ci = np.percentile(v, [2.5, 97.5])
print("precision gain ML-baseline CI", ci.round(3))


# --- which feature groups matter ---
G = dict(
    History=["past", "recent", "n_years"],
    Neighbors=["neighbors"],
    Coverage=["src1", "state_trend"],
    Road=["interstate"] + tags,
    Conditions=["night", "wet", "peak"],
)
imp = {}
for g, c in G.items():
    r = [x for x in feats if x not in c]
    q = pd.Series(
        H(max_depth=4, learning_rate=0.05, max_iter=200, random_state=0)
        .fit(Xa[r], ya)
        .predict_proba(Xb[r])[:, 1],
        index=Xb.index,
    )
    imp[g] = round(R["ML"]["ap"] - m(q)["ap"], 3)
print("drop-group AP loss", imp)


# --- california only ---
ca = Xb.state == "CA"
ks = int(yb[ca].sum())
t1 = ml[ca].sort_values(ascending=False).index[:ks]
t2 = base[ca].sort_values(ascending=False).index[:ks]
print(
    "CA positives",
    ks,
    "ML prec",
    round(yb.loc[t1].mean(), 3),
    "base prec",
    round(yb.loc[t2].mean(), 3),
)
json.dump(
    dict(k=k, n=len(Xb), rows=R, ci=[float(x) for x in ci], drop=imp),
    open(f"{OUT}/model_results.json", "w"),
    indent=1,
)
