# participant level plotting for parameter recovery

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as st

PROJECT_DIR = Path(os.getenv("PROJECT_DIR", "/workspace")).resolve()
FIG_DIR     = PROJECT_DIR / "figures_dir_garcia/garcia_replication_For_paper_6/recovery_For_paper_m6"
IN_CSV      = FIG_DIR / "true_vs_recovered_individual6.csv"
OUT_PNG     = FIG_DIR / "scatter_individual_ALL_params_6.png"
OUT_STATS   = FIG_DIR / "scatter_individual_ALL_params_6.csv"

df = pd.read_csv(IN_CSV).replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["true", "recovered", "parameter"])
if df.empty:
    raise SystemExit("Empty csv")

# order for plotting
priority = ['a',
            't',
            'z',
            'v_ES_AttentionW',
            'v_ES_InattentionW_E',
            'v_ES_InattentionW_S']
params = sorted(
    df["parameter"].unique(),
    key=lambda p: (p not in priority, priority.index(p) if p in priority else 0, p)
)

def regress_stats(x, y):
    x = np.asarray(x); y = np.asarray(y)
    n = len(x)
    ok = (n >= 3) and np.isfinite(x).all() and np.isfinite(y).all() and (np.std(x) > 0)
    if not ok:
        return dict(ok=False, N=n, R2=np.nan, p=np.nan, RMSE=np.nan)
    res  = st.linregress(x, y)
    yhat = res.intercept + res.slope * x
    rmse = float(np.sqrt(np.mean((y - yhat)**2)))
    r2   = float(res.rvalue**2)
    return dict(ok=True, N=n, R2=r2, p=float(res.pvalue), RMSE=rmse)


AX_LIMS = {"t": (0.0, 1.0)}

def _p_text(p):
    if not np.isfinite(p):
        return "p=NA"
    return "p<.001" if p < 1e-3 else f"p={p:.3f}"


def facet_scatter(data, **k):
    ax = plt.gca()
    x = data["true"].to_numpy()
    y = data["recovered"].to_numpy()
    pname = str(data["parameter"].iloc[0])
    ax.scatter(x, y, s=18, alpha=0.7)
    if pname in AX_LIMS:
        lo, hi = AX_LIMS[pname]
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        mask = (x >= lo) & (x <= hi) & (y >= lo) & (y <= hi)
        x_s, y_s = x[mask], y[mask]
    else:
        x_s, y_s = x, y

    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    lo_line, hi_line = min(x0, y0), max(x1, y1)
    ax.plot([lo_line, hi_line], [lo_line, hi_line], "--", lw=1, color="k")
    s = regress_stats(x_s, y_s)
    txt = (f"R²={s['R2']:.2f}\n{_p_text(s['p'])}\nRMSE={s['RMSE']:.3f}") if s["ok"] else f"N={s['N']}"
    ax.text(
        0.97, 0.03, txt,
        transform=ax.transAxes,
        ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9)
    )
    
    ax.set_xlabel("true value")
    ax.set_ylabel("posterior mean (recovered)")


rows = []
for p in params:
    sub = df[df["parameter"] == p]
    s = regress_stats(sub["true"], sub["recovered"])
    rows.append({"parameter": p, "R2": s["R2"], "p": s["p"], "RMSE": s["RMSE"]})
pd.DataFrame(rows, columns=["parameter","R2","p","RMSE"]).to_csv(OUT_STATS, index=False)

sns.set_style("white")
col_wrap = 3 if len(params) <= 9 else 4

g = sns.FacetGrid(df, col="parameter", col_order=params, col_wrap=col_wrap,
                  sharex=False, sharey=False, height=3.1, despine=True)
g.set_titles("{col_name}")
g.map_dataframe(facet_scatter)
g.figure.subplots_adjust(top=0.92)
g.figure.suptitle("Individual-level parameter recovery (R², p, RMSE)")
g.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
print(f"Saved figure: {OUT_PNG}")
print(f"Saved stats:  {OUT_STATS}")

