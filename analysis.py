import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

pd.set_option("display.width", 220)


# --- settings ---
CSV = sys.argv[1]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
SCALE = 7_728_394 / 500_000
ANNUITY = (1 - 1.07**-20) / 0.07
GRID = 0.01


# --- crash values (NHTSA 2022 mix, USDOT values) ---
F_, I_, P_ = 39221, 1664598, 4226677
K, C, O = 13_700_000, 122_400, 5_500
V_LOW = (I_ * C + P_ * O) / (F_ + I_ + P_)
V_HIGH = (F_ * K + I_ * C + P_ * O) / (F_ + I_ + P_)
HAIRCUT = 0.5
assert round(V_LOW) == 38276 and round(V_HIGH) == 128880
CMF = {"Intersection redesign": 0.44, "Friction / drainage resurfacing": 0.52}


# --- load and clean ---
df = pd.read_csv(CSV, low_memory=False)
df["st"] = pd.to_datetime(df.Start_Time, format="mixed")
df["et"] = pd.to_datetime(df.End_Time, format="mixed")
df["yr"] = df.st.dt.year
desc = df.Description.fillna("")
noncrash = desc.str.contains(r"^\s*(?:construction|road ?work)", case=False, regex=True)
print(
    "rows",
    len(df),
    "| non-crash construction events dropped",
    int(noncrash.sum()),
    "| 2023 partial-year rows dropped",
    int((df.yr == 2023).sum()),
)
print("by year\n", pd.crosstab(df.yr, df.Source))
df = df[~noncrash & (df.yr <= 2022)].copy()
print(
    "SEVERITY TRAP: share Sev3+ by source",
    (df.assign(s=df.Severity >= 3).groupby("Source").s.mean() * 100).round(1).to_dict(),
)


# --- road type and crash flags ---
s = df.Street.fillna("")
d = df.Description.fillna("")
df["road"] = np.select(
    [
        s.str.contains(r"\bI-\d|Interstate", regex=True),
        s.str.contains(
            r"\b(?:US|SR|CA|FL|TX|State|Hwy|Highway|NY|NC|SC|VA|PA)[- ]\d|Fwy|Expy|Tpke|Pkwy",
            regex=True,
        ),
    ],
    ["Interstate", "State/US/Fwy"],
    "Local",
)
df["night"] = df.Civil_Twilight.eq("Night")
df["wet"] = df.Weather_Condition.fillna("").str.contains(
    "Rain|Drizzle|Storm|Snow|Sleet|Hail|Ice|Wintry|Mix", case=False, regex=True
)
df["intx"] = df.Traffic_Signal | df.Stop
df["ramp"] = df.Junction | d.str.contains(r"\bramp\b", case=False, regex=True)
print("road class share", df.road.value_counts(normalize=True).round(3).to_dict())


# --- group crashes into ~1 km sites ---
df["site"] = (
    np.round(df.Start_Lat / GRID).astype(int).astype(str)
    + "_"
    + np.round(df.Start_Lng / GRID).astype(int).astype(str)
)
F = ["night", "wet", "intx", "ramp"]
for f in F:
    df["e_" + f] = df.groupby(["State", "road", "Source"])[f].transform("mean")


def agg(x):
    return x.groupby("site").agg(
        n=("ID", "size"), **{f: (f, "sum") for f in F}, **{"e_" + f: ("e_" + f, "sum") for f in F}
    )


tr = df[df.yr <= 2020]
te = df[df.yr >= 2021]
A = agg(tr)
B = agg(te)
ALL = agg(df)


# --- 1. do hotspots last, and do the worst sites fall back ---
c = A.n.sort_values(ascending=False)
t = B.n
rows = []
for p in [0.01, 0.05]:
    top = c.index[: int(len(c) * p)]
    rows.append(
        dict(
            group=f"top {p*100:.0f}%",
            past=c.loc[top].sum() / c.sum() * 100,
            future=t.reindex(top).fillna(0).sum() / t.sum() * 100,
        )
    )
    print(
        f"top {p*100:.0f}% of 2016-20 sites ({len(top)}): {c.loc[top].sum()/c.sum()*100:.1f}% of 2016-20 crashes -> {t.reindex(top).fillna(0).sum()/t.sum()*100:.1f}% of 2021-22"
    )
pd.DataFrame(rows).to_csv(f"{OUT}/persist.csv", index=False)
top100 = c.index[:100]
rk = t.rank(ascending=False)
print(
    "of top100 2016-20 sites, still in top 1000 in 2021-22:",
    int((rk.reindex(top100) <= 1000).sum()),
)
j = A[["n"]].join(t.rename("fut"), how="left").fillna(0)
j = j[j.n >= 5]
b, a = np.polyfit(np.log(j.n), np.log(j.fut + 0.5), 1)
fit = lambda n: np.exp(a) * n**b - 0.5
print(f"RTM fit: future_2yr = {np.exp(a):.3f} * n^{b:.3f}; network ratio={len(te)/len(tr):.3f}")
j["bin"] = pd.cut(j.n, [5, 10, 20, 40, 80, 1000], right=False)
rt = j.groupby("bin", observed=True).agg(
    sites=("n", "size"), past=("n", "mean"), future=("fut", "mean")
)
rt["ratio"] = rt.future / rt.past
print(rt.round(2))
rt.loc["all"] = [0, 0, 0, len(te) / len(tr)]
rt.to_csv(f"{OUT}/regression_to_mean.csv")


# --- 2. does a night or wet pattern last ---
def flag(X, f, minn):
    a = X[X.n >= minn].copy()
    e = a["e_" + f]
    ph = (e / a.n).clip(0.001, 0.999)
    z = (a[f] - e) / np.sqrt(e * (1 - ph))
    a["p"] = 1 - stats.norm.cdf(z)
    m = len(a)
    a["q"] = (a.p * m / a.p.rank()).clip(upper=1)
    return a[(a.q < 0.05) & (a[f] / a.n >= 1.25 * e / a.n)]


for f in ["night", "wet"]:
    for minn in [10, 20]:
        fl = flag(A, f, minn)
        bb = B.reindex(fl.index).dropna()
        bb = bb[bb.n >= 5]
        base = A[A.n >= minn].drop(fl.index)
        bu = B.reindex(base.index).dropna()
        bu = bu[bu.n >= 5]
        if len(bb):
            print(
                f'{f} min{minn}: flagged {len(fl)}, testable {len(bb)}, past obs/exp {fl[f].sum()/fl["e_"+f].sum():.2f}, FUTURE obs/exp {bb[f].sum()/bb["e_"+f].sum():.2f} (crashes {int(bb.n.sum())}), unflagged future {bu[f].sum()/bu["e_"+f].sum():.2f}'
            )


# --- 3. forecast, diagnose and rank sites ---
elig = ALL.index[ALL.n >= 20]
ds = df[df.site.isin(elig)]
top = lambda x: (x.dropna().mode().iat[0] if x.notna().any() else "")
meta = ds.groupby("site").agg(
    state=("State", top),
    county=("County", top),
    city=("City", top),
    street=("Street", top),
    road=("road", top),
    lat=("Start_Lat", "mean"),
    lng=("Start_Lng", "mean"),
)
P = ALL.join(meta, how="inner")
P["n_past"] = A.n.reindex(P.index).fillna(0)
P["n_recent"] = B.n.reindex(P.index).fillna(0)
P = P[(P.n >= 20) & (P.n_recent >= 5)].copy()
P["annual_sample"] = (fit(P.n_past.clip(lower=1)).clip(lower=0) + P.n_recent) / 2 / 2
P["annual_crashes"] = P.annual_sample * SCALE
P["intx_share"] = P.intx / P.n
P["ramp_share"] = P.ramp / P.n
P["wet_share"] = P.wet / P.n
P["night_share"] = P.night / P.n
wetflag = flag(ALL, "wet", 20).index
P["wet_flag"] = P.index.isin(wetflag)
P["project"] = np.select(
    [P.wet_flag, P.intx_share >= 0.5, P.ramp_share >= 0.5],
    ["Friction / drainage resurfacing", "Intersection redesign", "Interchange / ramp study"],
    "Corridor study",
)
P["target_share"] = np.select(
    [
        P.project.str.startswith("Fric"),
        P.project.str.startswith("Inter") & (P.project != "Interchange / ramp study"),
    ],
    [P.wet_share, P.intx_share],
    0.0,
)
P["reduction"] = P.project.map(CMF).fillna(0.0)
P["crashes_avoided_yr"] = P.annual_crashes * P.target_share * P.reduction
P["benefit_low"] = P.crashes_avoided_yr * HAIRCUT * V_LOW * ANNUITY
P["benefit_high"] = P.crashes_avoided_yr * V_HIGH * ANNUITY
P["justified_budget"] = P.benefit_low
ti = df[df.site.isin(P.index) & df.intx & df.road.ne("Interstate")]
lab = ti.groupby("site").agg(t_street=("Street", top), t_road=("road", top), t_city=("City", top))
P = P.join(lab)
isx = P.project == "Intersection redesign"
P.loc[isx & P.t_street.notna(), "street"] = P.t_street
P.loc[isx & P.t_road.notna(), "road"] = P.t_road
P.loc[isx & P.t_city.notna(), "city"] = P.t_city
P["buyer"] = np.where(P.road == "Local", "City / County", "State DOT")
print(f"V_LOW {V_LOW:,.0f}  V_HIGH {V_HIGH:,.0f}  annuity {ANNUITY:.3f}  scale {SCALE:.2f}")
print("eligible sites", len(P))
print(
    P.groupby("project")
    .agg(
        sites=("n", "size"),
        low_m=("benefit_low", lambda x: x.sum() / 1e6),
        high_m=("benefit_high", lambda x: x.sum() / 1e6),
    )
    .round(1)
)
Q = P[P.reduction > 0].sort_values("benefit_low", ascending=False).copy()
Q["county_rank"] = Q.groupby(["state", "county"]).benefit_low.rank(ascending=False).astype(int)
cols = [
    "state",
    "county",
    "county_rank",
    "city",
    "street",
    "road",
    "buyer",
    "project",
    "n",
    "n_recent",
    "annual_crashes",
    "intx_share",
    "wet_share",
    "night_share",
    "crashes_avoided_yr",
    "benefit_low",
    "benefit_high",
    "lat",
    "lng",
]
Q[cols].round(3).to_csv(f"{OUT}/ranked_sites.csv", index_label="site")
print(
    Q.groupby("buyer")
    .agg(sites=("n", "size"), low_m=("benefit_low", lambda x: x.sum() / 1e6))
    .round(1)
)


# --- 4. southern california list ---
SC = Q[(Q.state == "CA") & Q.county.isin(["Los Angeles", "Orange", "San Bernardino", "Riverside"])]
print(
    "SoCal sites",
    len(SC),
    "low $M",
    round(SC.benefit_low.sum() / 1e6, 1),
    "high $M",
    round(SC.benefit_high.sum() / 1e6, 1),
    SC.groupby("buyer").benefit_low.sum().div(1e6).round(1).to_dict(),
    SC.groupby("county").size().to_dict(),
    SC.project.value_counts().to_dict(),
)
print(
    SC[
        [
            "county",
            "city",
            "street",
            "road",
            "buyer",
            "project",
            "n",
            "n_recent",
            "annual_crashes",
            "benefit_low",
            "benefit_high",
        ]
    ]
    .head(10)
    .round(0)
    .to_string()
)
SC[cols].round(3).to_csv(f"{OUT}/socal_pipeline.csv", index_label="site")
t10 = SC.index[:10]
x = df[df.site.isin(t10) & df.intx].copy()
x["hr"] = x.st.dt.hour
x["wk"] = x.st.dt.dayofweek < 5
print("case-study top10 intersection crashes", len(x))
h = x[x.wk].hr.value_counts(normalize=True).sort_index() * 100
bl = {
    "overnight 22-05": [22, 23, 0, 1, 2, 3, 4],
    "am peak 6-9": [6, 7, 8],
    "midday 9-15": list(range(9, 15)),
    "pm peak 15-19": list(range(15, 19)),
    "evening 19-22": [19, 20, 21],
}
print(
    {
        k: (round(h.reindex(v).fillna(0).sum(), 1), round(h.reindex(v).fillna(0).sum() / len(v), 1))
        for k, v in bl.items()
    }
)
top1 = SC.index[0]
y = df[(df.site == top1) & df.intx]
print("TOP SITE", top1, SC.loc[top1, ["lat", "lng"]].round(4).to_dict())
print(y.Street.value_counts().head(5).to_dict())
print(y.Description.sample(6, random_state=3).tolist())
print(y.yr.value_counts().sort_index().to_dict(), y.Source.value_counts().to_dict())
