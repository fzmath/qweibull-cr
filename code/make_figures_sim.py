# -*- coding: utf-8 -*-
"""仿真结果图: Fig4 RMSE/覆盖率 (MLE vs Bayes); Fig5 效率 (NUTS vs MH)。"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 13, "axes.edgecolor": "black", "axes.linewidth": 1.2,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "axes.axisbelow": True, "figure.dpi": 150})

ROOT = r"G:\OPT\q-Weibull竞争风险论文"
sim = json.load(open(ROOT + r"\data\sim_results_heavy.json", encoding="utf-8"))
labels = ["Strong\ntruncation", "Moderate\ntruncation", "Weak\ntruncation"]
keys = ["强截断(竞争早)", "中截断", "弱截断(竞争晚)"]
x = np.arange(3); w = 0.36

mle_rmse = [sim[k]["mle_rmse"] for k in keys]
bay_rmse = [sim[k]["bay_rmse"] for k in keys]
mle_cov = [sim[k]["mle_cover"] for k in keys]
bay_cov = [sim[k]["bay_cover"] for k in keys]

# ---- Fig4: RMSE + coverage ----
fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6))
a.bar(x - w/2, mle_rmse, w, label="MLE", color="#9aa7b8", edgecolor="black", lw=.8)
a.bar(x + w/2, bay_rmse, w, label="Bayes (NUTS)", color="#2f6db0",
      edgecolor="black", lw=.8)
a.set_xticks(x); a.set_xticklabels(labels)
a.set_ylabel("RMSE of $q$"); a.set_title("(a) Root-mean-squared error")
a.legend(frameon=False)
for xi, v in zip(x - w/2, mle_rmse):
    a.text(xi, v + .01, f"{v:.2f}", ha="center", fontsize=10)
for xi, v in zip(x + w/2, bay_rmse):
    a.text(xi, v + .01, f"{v:.2f}", ha="center", fontsize=10)

b.plot(x, mle_cov, "o--", color="#6b7785", label="MLE (Wald)", lw=1.8, ms=8)
b.plot(x, bay_cov, "s-", color="#2f6db0", label="Bayes (95% CI)", lw=1.8, ms=8)
b.axhline(0.95, color="crimson", ls=":", lw=1.6, label="Nominal 0.95")
b.set_xticks(x); b.set_xticklabels(labels)
b.set_ylabel("95% interval coverage"); b.set_ylim(0, 1.05)
b.set_title("(b) Coverage of true $q$"); b.legend(frameon=False, fontsize=11)
fig.tight_layout()
fig.savefig(ROOT + r"\figures\fig4_sim_performance.png", bbox_inches="tight")
fig.savefig(ROOT + r"\figures\fig4_sim_performance.pdf", bbox_inches="tight")
print("Fig4 saved")

# ---- Fig5: efficiency ----
try:
    eff = json.load(open(ROOT + r"\data\efficiency_results.json", encoding="utf-8"))
    enames = list(eff.keys())
    nuts_es = [eff[k]["nuts_ess_sec"] for k in enames]
    mh_es = [eff[k]["mh_ess_sec"] for k in enames]
    sp = [eff[k]["speedup"] for k in enames]
    xe2 = np.arange(len(enames))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(xe2 - w/2, mh_es, w, label="Random-walk MH (preconditioned)",
           color="#c98a5b", edgecolor="black", lw=.8)
    ax.bar(xe2 + w/2, nuts_es, w, label="NUTS", color="#2f6db0",
           edgecolor="black", lw=.8)
    ax.set_xticks(xe2); ax.set_xticklabels(enames)
    ax.set_ylabel("Total effective sample size per second (ESS/s)")
    ax.set_title("Computational efficiency: NUTS vs random-walk Metropolis")
    for xi, v in zip(xe2 + w/2, nuts_es):
        ax.text(xi, v + max(nuts_es)*.01, f"{v:.0f}", ha="center", fontsize=10)
    for xi, v in zip(xe2 - w/2, mh_es):
        ax.text(xi, v + max(nuts_es)*.01, f"{v:.0f}", ha="center", fontsize=10)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT + r"\figures\fig5_efficiency.png", bbox_inches="tight")
    fig.savefig(ROOT + r"\figures\fig5_efficiency.pdf", bbox_inches="tight")
    print("Fig5 saved; speedups=", [f"{s:.1f}x" for s in sp])
except FileNotFoundError:
    print("efficiency_results.json 尚未生成, 跳过 Fig5")
