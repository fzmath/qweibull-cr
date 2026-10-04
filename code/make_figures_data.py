# -*- coding: utf-8 -*-
"""真实数据论文图 (矢量PDF + 300dpi PNG):
Fig1 数据时间轴; Fig2 q 后验密度; Fig3 CIF/R_sys + Kaplan-Meier 对照。"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11, "axes.linewidth": 0.9, "figure.dpi": 130})
OUT = r"G:\OPT\q-Weibull竞争风险论文\figures"

df = pd.read_csv(r"G:\OPT\q-Weibull竞争风险论文\data\voltage.csv")
t = df["hours"].to_numpy(float); fm = df["failure_mode"].to_numpy(str)
st = df["status"].to_numpy(int)
col = {"E": "#1f6fb4", "D": "#d14949", "censored": "#6b6b6b"}

# ---------- Fig1 数据时间轴 ----------
fig, ax = plt.subplots(figsize=(7, 2.6))
groups = [("E", 2), ("D", 1), ("censored", 0)]
for lab, y in groups:
    tt = t[fm == lab]
    ax.scatter(tt, np.full(len(tt), y), s=22, color=col[lab],
               marker=("x" if lab == "censored" else "o"),
               label=("E (early)" if lab == "E" else
                      "D (degradation)" if lab == "D" else "censored"))
ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["censored", "D", "D" if False else "E"])
ax.set_yticklabels(["censored", "D", "E"])
ax.set_xlabel("Time (hours)"); ax.set_ylim(-0.6, 2.6)
ax.legend(ncol=3, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.28))
ax.grid(axis="x", ls=":", alpha=0.5)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_data.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/fig1_data.png", bbox_inches="tight", dpi=300)
plt.close(fig)

# ---------- Fig2 q 后验密度 ----------
ch = np.load(r"G:\OPT\q-Weibull竞争风险论文\data\post_chains.npy")
post = ch[:, 1000:, :].reshape(-1, 6)
qE = 2-np.exp(post[:, 0]); qD = 2-np.exp(post[:, 3])
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8), sharey=True)
for ax, qq, ttl in [(axes[0], qE, "E (early)"), (axes[1], qD, "D (degradation)")]:
    xs = np.linspace(0.4, 1.8, 300)
    kd = gaussian_kde(qq, 0.35)
    ax.fill_between(xs, kd(xs), color="#7fa8c9", alpha=0.5)
    ax.plot(xs, kd(xs), color="#1f6fb4", lw=1.4)
    ax.axvline(1.0, color="k", ls="--", lw=1.1)
    ax.set_title(ttl); ax.set_xlabel("q")
    ax.set_xlim(0.4, 1.8); ax.grid(ls=":", alpha=0.4)
axes[0].set_ylabel("Posterior density")
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_qpost.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/fig2_qpost.png", bbox_inches="tight", dpi=300)
plt.close(fig)

# ---------- Fig3 CIF / R_sys + KM ----------
Q = np.load(r"G:\OPT\q-Weibull竞争风险论文\data\posterior_quantities.npz")
tg = Q["tg"]
def km_curve(tt, ss):
    o = np.argsort(tt); tt, ss = tt[o], ss[o]
    times, S, n = [0], [1.0], len(tt)
    i = 0
    while i < n:
        ti = tt[i]
        if ss[i] == 1:
            d = np.sum((tt == ti) & (ss == 1)); risk = np.sum(tt >= ti)
            S.append(S[-1]*(1-d/risk)); times.append(ti)
        i += 1
    return times, S
kmx, kmy = km_curve(t, st)

fig, ax = plt.subplots(figsize=(6.6, 4.2))
m = tg <= 700
ax.fill_between(tg[m], Q["rsys_lo"][m], Q["rsys_hi"][m], color="#2e7d32", alpha=0.15)
ax.plot(tg[m], Q["rsys_mean"][m], color="#2e7d32", lw=1.8, label="System reliability $R_{sys}$")
ax.step(kmx, kmy, where="post", color="k", ls=":", lw=1.4, label="Kaplan-Meier")
for j, cc, ttl in [("e", "#1f6fb4", "CIF: E"), ("d", "#d14949", "CIF: D")]:
    ax.fill_between(tg[m], Q[f"cif_{j}_lo"][m], Q[f"cif_{j}_hi"][m],
                    color=cc, alpha=0.12)
    ax.plot(tg[m], Q[f"cif_{j}_mean"][m], color=cc, lw=1.6, ls="--", label=ttl)
ax.set_xlabel("Time (hours)"); ax.set_ylabel("Probability")
ax.set_xlim(0, 700); ax.set_ylim(0, 1.02); ax.grid(ls=":", alpha=0.5)
ax.legend(frameon=False, loc="center right")
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_cif.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/fig3_cif.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("真实数据图已生成 fig1-3")
