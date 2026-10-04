# -*- coding: utf-8 -*-
"""重做 fig4 (多 q RMSE 有效区域) 与 fig5 (4 采样器 benchmark), png+pdf。"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = r"G:\OPT\q-Weibull竞争风险论文"
GD = os.path.join(ROOT, "data", "sim_grid")
FIG = os.path.join(ROOT, "figures")
QS = [0.6, 0.8, 1.0, 1.25, 1.45, 1.7]
SC = ["strong", "moderate", "weak"]
SCL = ["Strong", "Moderate", "Weak"]

R = np.zeros((len(QS), 3)); RM = np.zeros((len(QS), 3)); RB = np.zeros((len(QS), 3))
for i, q in enumerate(QS):
    for j, s in enumerate(SC):
        d = json.load(open(os.path.join(GD, f"sim_q{q}_{s}.json"), encoding="utf-8"))
        R[i, j] = d["bayes"]["rmse"]/d["mle"]["rmse"]
        RM[i, j] = d["mle"]["rmse"]; RB[i, j] = d["bayes"]["rmse"]

# ---------- fig4 ----------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
norm = TwoSlopeNorm(vcenter=1.0, vmin=0.2, vmax=1.8)
im = ax[0].imshow(R, cmap="RdBu", norm=norm, aspect="auto")
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(SCL)
ax[0].set_yticks(range(len(QS))); ax[0].set_yticklabels([f"q={q}" for q in QS])
for i in range(len(QS)):
    for j in range(3):
        ax[0].text(j, i, f"{R[i,j]:.2f}", ha="center", va="center", fontsize=10,
                   color="black")
ax[0].set_title("(a) RMSE ratio  Bayes / MLE", fontsize=11)
fig.colorbar(im, ax=ax[0], shrink=0.8, label="ratio (<1: Bayes better)")

x = np.arange(len(QS)); w = 0.38
ax[1].bar(x-w/2, RM[:, 0], w, label="MLE", color="#c0504d")
ax[1].bar(x+w/2, RB[:, 0], w, label="Bayes (NUTS)", color="#4f81bd")
ax[1].set_xticks(x); ax[1].set_xticklabels([str(q) for q in QS])
ax[1].set_xlabel("true q"); ax[1].set_ylabel("RMSE of q")
ax[1].set_title("(b) Strong truncation: RMSE by q", fontsize=11)
ax[1].legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig4_sim_performance.png"), dpi=300)
fig.savefig(os.path.join(FIG, "fig4_sim_performance.pdf"))
plt.close(fig)

# ---------- fig5 ----------
b = json.load(open(os.path.join(ROOT, "data", "benchmark_results.json"), encoding="utf-8"))
order = ["RW-MH", "Adaptive-MH", "HMC(precond)", "NUTS"]
labs = ["RW-MH", "Adaptive-MH", "HMC (precond.)", "NUTS"]
esss = [b[k]["ess_sec"] for k in order]
essg = [b["HMC(precond)"]["ess_grad"], b["NUTS"]["ess_grad"]]

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
bars = ax[0].bar(labs, esss, color=["#9bbb59", "#8db4e2", "#f79646", "#4f81bd"])
ax[0].set_ylabel("effective samples / second")
ax[0].set_title("(a) Sampling efficiency (wall-clock)", fontsize=11)
for r, v in zip(bars, esss):
    ax[0].text(r.get_x()+r.get_width()/2, v+1.5, f"{v:.0f}", ha="center", fontsize=9)
ax[0].tick_params(axis="x", labelrotation=20)

bars2 = ax[1].bar(["HMC (fixed L)", "NUTS"], essg, color=["#f79646", "#4f81bd"],
                  width=0.5)
ax[1].set_ylabel("effective samples / gradient evaluation")
ax[1].set_title("(b) Gradient utilization", fontsize=11)
for r, v in zip(bars2, essg):
    ax[1].text(r.get_x()+r.get_width()/2, v+0.004, f"{v:.3f}", ha="center",
               fontsize=10)
ax[1].set_ylim(0, 0.22)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig5_efficiency.png"), dpi=300)
fig.savefig(os.path.join(FIG, "fig5_efficiency.pdf"))
plt.close(fig)
print("fig4, fig5 regenerated")
