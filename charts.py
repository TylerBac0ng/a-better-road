import json, os
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
out = here + "/outputs/"
os.makedirs(here + "/charts", exist_ok=True)
yellow = "#EDC932"
blue = "#2E8BC9"
gray = "#AEB3BC"
grid = "#6B7488"
white = "#FFFFFF"
plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 12,
        "text.color": gray,
        "axes.labelcolor": gray,
        "xtick.color": gray,
        "ytick.color": gray,
    }
)


def setup():
    f, a = plt.subplots(figsize=(6, 3.84))
    f.patch.set_alpha(0)
    a.patch.set_alpha(0)
    for s in ["top", "right", "left"]:
        a.spines[s].set_visible(False)
    a.spines["bottom"].set_color(grid)
    a.yaxis.grid(True, color=grid, lw=1)
    a.set_axisbelow(True)
    a.tick_params(axis="y", length=0)
    a.tick_params(axis="x", length=0)
    return f, a


def vals(a, bars, fmt):
    for b in bars:
        a.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + a.get_ylim()[1] * 0.012,
            fmt.format(b.get_height()),
            ha="center",
            color=white,
            fontsize=12,
            fontweight="bold",
        )


def save(f, name):
    f.savefig(here + "/charts/" + name, dpi=300, transparent=True)
    plt.close(f)


# --- hotspots ---
p = pd.read_csv(out + "persist.csv")
f, a = setup()
a.set_ylim(0, 40)
a.set_yticks(range(0, 41, 5), [f"{i}%" for i in range(0, 41, 5)])
w = 0.35
b1 = a.bar([-w / 2, 1 - w / 2], p.past, w, color=yellow, label="2016–20 (used to pick sites)")
b2 = a.bar([w / 2, 1 + w / 2], p.future, w, color=blue, label="2021–22 (later test)")
vals(a, list(b1) + list(b2), "{:.1f}")
a.set_xticks([0, 1], ["Top 1% of sites", "Top 5% of sites"])
a.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1),
    ncol=2,
    frameon=False,
    handlelength=1,
    handleheight=1,
    labelcolor=gray,
)
f.subplots_adjust(left=0.09, right=0.98, top=0.9, bottom=0.09)
save(f, "hotspots.png")


# --- regression to the mean ---
r = pd.read_csv(out + "regression_to_mean.csv")
all_ = r[r["bin"] == "all"].ratio.iat[0]
r = r[r["bin"] != "all"]
v = [all_ * 100] + list(r.ratio * 100)
f, a = setup()
a.set_ylim(0, 100)
a.set_yticks(range(0, 101, 20), [f"{i}%" for i in range(0, 101, 20)])
b = a.bar(range(6), v, 0.5, color=[blue] + [yellow] * 5)
vals(a, b, "{:.0f}")
a.set_xticks(range(6), ["All sites", "5–9", "10–19", "20–39", "40–79", "80+"])
a.set_xlabel("Past crash count at the site", labelpad=6)
f.subplots_adjust(left=0.09, right=0.98, top=0.96, bottom=0.16)
save(f, "rtm.png")


# --- model ---
m = json.load(open(out + "model_results.json"))
rows = [
    ("Random pick", "Random", gray),
    ("Past crash count", "Past count", blue),
    ("ML model", "ML", yellow),
]
f, a = setup()
a.set_ylim(0, 50)
a.set_yticks(range(0, 51, 10), [f"{i}%" for i in range(0, 51, 10)])
b = a.bar(
    range(3),
    [m["rows"][k]["precision"] * 100 for _, k, _c in rows],
    0.5,
    color=[c for *_, c in rows],
)
vals(a, b, "{:.1f}")
a.set_xticks(range(3), [l for l, *_ in rows])
f.subplots_adjust(left=0.09, right=0.98, top=0.96, bottom=0.09)
save(f, "model.png")
