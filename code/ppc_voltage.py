# -*- coding: utf-8 -*-
"""P1-9 后验预测检验 (PPC) for voltage q-Weibull 竞争风险模型。

对每个后验 draw 生成 replicate 数据集 (n=58, 固定观测删失方案),
汇总 5 类统计量与观测数据比较。
输出: data/ppc.json, figures/fig10_ppc.pdf/.png
"""
import sys, os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
from qweibull import sf, pdf

# ----------------------------------------------------------------------
# 1. 加载观测数据
# ----------------------------------------------------------------------
df = pd.read_csv(r"G:\OPT\q-Weibull竞争风险论文\data\voltage.csv")
t_obs = df["hours"].to_numpy(float)
fm = df["failure_mode"].to_numpy(str)
n = len(t_obs)
cause_obs = np.full(n, -1, dtype=int)   # -1=censored, 0=E, 1=D
cause_obs[fm == "E"] = 0
cause_obs[fm == "D"] = 1
event_obs = (cause_obs >= 0).astype(int)  # 1=失效, 0=删失

nE_obs = int((cause_obs == 0).sum())
nD_obs = int((cause_obs == 1).sum())
max_t_obs = float(t_obs.max())

print(f"观测: n={n}, E={nE_obs}, D={nD_obs}, censored={n-nE_obs-nD_obs}, max(t)={max_t_obs}")

# 固定删失时间: 删失观测的观测时间作为 C_i, 失效观测 C_i=inf
C = np.full(n, np.inf)
C[cause_obs == -1] = t_obs[cause_obs == -1]

# ----------------------------------------------------------------------
# 2. 加载后验链, 抽样子样本
# ----------------------------------------------------------------------
ch = np.load(r"G:\OPT\q-Weibull竞争风险论文\data\post_chains.npy")  # (2, 2000, 6)
post = ch[:, 1000:, :].reshape(-1, 6)  # (2000, 6)
rng = np.random.default_rng(2026)
B = 300
idx = rng.choice(len(post), size=B, replace=False)
draws = post[idx]  # (B, 6)
print(f"后验 draw 数: {B} (从 {len(post)} 个 post-warmup 抽样)")

# ----------------------------------------------------------------------
# 3. 工具函数: q-Weibull 逆 CDF 采样
# ----------------------------------------------------------------------
def rnd_qw(q, eta, beta, m, rng):
    u = np.maximum(rng.random(m), 1e-12)
    lam = eta ** (-beta)
    if abs(q - 1.0) < 1e-9:
        return eta * (-np.log(u)) ** (1.0 / beta)
    d = 1.0 - q
    return ((1.0 - u ** (1.0 / ((2.0 - q) / d))) / (d * lam)) ** (1.0 / beta)

# ----------------------------------------------------------------------
# 4. KM 和 Aalen-Johansen CIF 估计量
# ----------------------------------------------------------------------
def km_curve(t, event, grid):
    """在 grid 上评估 KM 生存曲线。"""
    order = np.argsort(t)
    ts = t[order]
    es = event[order]
    surv = 1.0
    surv_pts = [1.0]
    time_pts = [0.0]
    i = 0
    nt = len(ts)
    for g in grid:
        while i < nt and ts[i] <= g:
            if es[i] == 1:
                n_risk = (ts >= ts[i]).sum()
                n_death = ((ts == ts[i]) & (es == 1)).sum()
                surv *= (1.0 - n_death / n_risk)
            i += 1
        surv_pts.append(surv)
        time_pts.append(g)
    return np.array(surv_pts)

def cif_aj(t, event, cause, target_cause, grid):
    """Aalen-Johansen CIF for target_cause (0 or 1), evaluated at grid."""
    order = np.argsort(t)
    ts = t[order]
    es = event[order]
    cs = cause[order]
    nt = len(ts)
    # KM survival at each event time
    cif_val = 0.0
    km_surv_before = 1.0  # KM survival just before current time
    cif_pts = [0.0]
    i = 0
    # Build event-time list
    for g in grid:
        while i < nt and ts[i] <= g:
            if es[i] == 1:
                n_risk = (ts >= ts[i]).sum()
                # count events at this time
                mask = (ts == ts[i])
                n_death_all = mask.sum()
                n_death_cause = ((cs == target_cause) & mask).sum()
                # contribution to CIF: S(t-) * d_cause / n_risk
                cif_val += km_surv_before * n_death_cause / n_risk
                # update KM survival after this event time
                km_surv_before *= (1.0 - n_death_all / n_risk)
            i += 1
        cif_pts.append(cif_val)
    return np.array(cif_pts)

# ----------------------------------------------------------------------
# 5. 逐 draw 生成 replicate 并汇总
# ----------------------------------------------------------------------
# 时间网格用于 KM / CIF 曲线
t_grid = np.linspace(0, 460, 200)
# 关注时刻
t_cif_target = 446.0
t_sys_points = np.array([50.0, 100.0, 200.0, 300.0, 400.0])

# 存储
km_rep = np.zeros((B, len(t_grid) + 1))
cif_e_rep = np.zeros((B, len(t_grid) + 1))
cif_d_rep = np.zeros((B, len(t_grid) + 1))
nE_rep = np.zeros(B, dtype=int)
nD_rep = np.zeros(B, dtype=int)
max_rep = np.zeros(B)
cif_e_446_rep = np.zeros(B)
cif_d_446_rep = np.zeros(B)
rsys_at_pts = np.zeros((B, len(t_sys_points)))

for b in range(B):
    x = draws[b]
    q_E = 2.0 - np.exp(x[0]); eta_E = np.exp(x[1]); beta_E = np.exp(x[2])
    q_D = 2.0 - np.exp(x[3]); eta_D = np.exp(x[4]); beta_D = np.exp(x[5])

    # 生成潜失效时间
    T0 = rnd_qw(q_E, eta_E, beta_E, n, rng)
    T1 = rnd_qw(q_D, eta_D, beta_D, n, rng)

    t_latent = np.minimum(T0, T1)
    cause_latent = (T1 < T0).astype(int)  # 0=E(T0 wins), 1=D(T1 wins)

    # 应用固定删失
    t_rep = np.minimum(t_latent, C)
    event_rep = (t_latent < C).astype(int)
    cause_rep = np.where(event_rep == 1, cause_latent, -1)

    # (b) 计数
    nE_rep[b] = int((cause_rep == 0).sum())
    nD_rep[b] = int((cause_rep == 1).sum())

    # (c) 最大观测寿命
    max_rep[b] = float(t_rep.max())

    # (a) KM 曲线
    km_rep[b] = km_curve(t_rep, event_rep, t_grid)

    # (d) CIF at 446 (A-J from replicate data)
    cif_e_rep[b] = cif_aj(t_rep, event_rep, cause_rep, 0, t_grid)
    cif_d_rep[b] = cif_aj(t_rep, event_rep, cause_rep, 1, t_grid)

    i446 = np.argmin(np.abs(t_grid - t_cif_target))
    cif_e_446_rep[b] = cif_e_rep[b, i446 + 1]  # +1 because grid starts with 0 at index 0
    cif_d_446_rep[b] = cif_d_rep[b, i446 + 1]

    # (e) system survival at observation time points
    for j, tp in enumerate(t_sys_points):
        i_tp = np.argmin(np.abs(t_grid - tp))
        rsys_at_pts[b, j] = km_rep[b, i_tp + 1]

# ----------------------------------------------------------------------
# 6. 观测值的对应统计量
# ----------------------------------------------------------------------
km_obs = km_curve(t_obs, event_obs, t_grid)
cif_e_obs_curve = cif_aj(t_obs, event_obs, cause_obs, 0, t_grid)
cif_d_obs_curve = cif_aj(t_obs, event_obs, cause_obs, 1, t_grid)

i446 = np.argmin(np.abs(t_grid - t_cif_target))
cif_e_446_obs = float(cif_e_obs_curve[i446 + 1])
cif_d_446_obs = float(cif_d_obs_curve[i446 + 1])

rsys_obs_pts = np.zeros(len(t_sys_points))
for j, tp in enumerate(t_sys_points):
    i_tp = np.argmin(np.abs(t_grid - tp))
    rsys_obs_pts[j] = km_obs[i_tp + 1]

# ----------------------------------------------------------------------
# 7. PPC 汇总函数
# ----------------------------------------------------------------------
def ppc_summary(obs_val, rep_samples, name):
    med = float(np.median(rep_samples))
    lo = float(np.percentile(rep_samples, 2.5))
    hi = float(np.percentile(rep_samples, 97.5))
    covered = bool(lo <= obs_val <= hi)
    return {
        "name": name,
        "observed": float(obs_val),
        "ppc_median": med,
        "ppc_95_lo": lo,
        "ppc_95_hi": hi,
        "covered": covered,
    }

results = {}
results["nE"] = ppc_summary(nE_obs, nE_rep, "E failure count")
results["nD"] = ppc_summary(nD_obs, nD_rep, "D failure count")
results["max_t"] = ppc_summary(max_t_obs, max_rep, "max observed lifetime")
results["cif_E_446"] = ppc_summary(cif_e_446_obs, cif_e_446_rep, "CIF_E(446h)")
results["cif_D_446"] = ppc_summary(cif_d_446_obs, cif_d_446_rep, "CIF_D(446h)")

for j, tp in enumerate(t_sys_points):
    results[f"rsys_{int(tp)}h"] = ppc_summary(
        rsys_obs_pts[j], rsys_at_pts[:, j], f"system survival at {int(tp)}h")

results["meta"] = {
    "n_draws": int(B),
    "n_subjects": int(n),
    "censoring_approach": "fixed observed censoring times applied to posterior predictive draws",
    "t_grid_max": float(t_grid[-1]),
}

# 打印汇总表
print("\n===== PPC Summary =====")
print(f"{'Statistic':<25} {'Observed':>10} {'PPC median':>12} {'95% interval':>20} {'Covered':>8}")
for k, v in results.items():
    if k == "meta":
        continue
    print(f"{v['name']:<25} {v['observed']:>10.4f} {v['ppc_median']:>12.4f} "
          f"[{v['ppc_95_lo']:>8.4f}, {v['ppc_95_hi']:>8.4f}]  {'YES' if v['covered'] else 'NO':>6}")

# 写 JSON
out_json = r"G:\OPT\q-Weibull竞争风险论文\data\ppc.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n已写 {out_json}")

# ----------------------------------------------------------------------
# 8. 画图: fig10_ppc.pdf + .png
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# --- Panel 1: KM 观测 vs PPC band ---
ax = axes[0, 0]
km_med = np.median(km_rep, axis=0)
km_lo = np.percentile(km_rep, 2.5, axis=0)
km_hi = np.percentile(km_rep, 97.5, axis=0)
ax.fill_between(t_grid, km_lo[1:], km_hi[1:], alpha=0.3, color="steelblue",
                label="PPC 95% band")
ax.plot(t_grid, km_med[1:], color="steelblue", lw=2, label="PPC median")
ax.step(t_grid, km_obs[1:], where="post", color="black", lw=2, label="Observed KM")
ax.set_xlabel("Time (hours)")
ax.set_ylabel("Survival probability")
ax.set_title("(a) System Survival: Observed vs PPC")
ax.legend(fontsize=9)
ax.set_xlim(0, 460)
ax.set_ylim(0, 1.05)

# --- Panel 2: E/D 计数直方图 ---
ax = axes[0, 1]
bins = np.arange(5, 45) - 0.5
ax.hist(nE_rep, bins=bins, alpha=0.6, color="crimson", label=f"E failures (obs={nE_obs})")
ax.hist(nD_rep, bins=bins, alpha=0.6, color="navy", label=f"D failures (obs={nD_obs})")
ax.axvline(nE_obs, color="crimson", ls="--", lw=2)
ax.axvline(nD_obs, color="navy", ls="--", lw=2)
ax.set_xlabel("Number of failures")
ax.set_ylabel("Posterior predictive frequency")
ax.set_title("(b) Failure Counts by Cause")
ax.legend(fontsize=9)

# --- Panel 3: max(t) + CIF at 446 ---
ax = axes[1, 0]
ax.hist(max_rep, bins=25, alpha=0.6, color="forestgreen", edgecolor="k")
ax.axvline(max_t_obs, color="black", ls="--", lw=2, label=f"Observed max(t)={max_t_obs:.0f}h")
ax.set_xlabel("Max observed lifetime (hours)")
ax.set_ylabel("Posterior predictive frequency")
ax.set_title("(c) Maximum Observed Lifetime")
ax.legend(fontsize=9)

# --- Panel 4: CIF at 446 ---
ax = axes[1, 1]
ax.hist(cif_e_446_rep, bins=20, alpha=0.6, color="crimson",
        label=f"CIF_E(446) obs={cif_e_446_obs:.3f}")
ax.hist(cif_d_446_rep, bins=20, alpha=0.6, color="navy",
        label=f"CIF_D(446) obs={cif_d_446_obs:.3f}")
ax.axvline(cif_e_446_obs, color="crimson", ls="--", lw=2)
ax.axvline(cif_d_446_obs, color="navy", ls="--", lw=2)
ax.set_xlabel("CIF at 446 hours")
ax.set_ylabel("Posterior predictive frequency")
ax.set_title("(d) Cumulative Incidence at 446h")
ax.legend(fontsize=9)

plt.tight_layout()
figpath_pdf = r"G:\OPT\q-Weibull竞争风险论文\figures\fig10_ppc.pdf"
figpath_png = r"G:\OPT\q-Weibull竞争风险论文\figures\fig10_ppc.png"
plt.savefig(figpath_pdf, dpi=300, bbox_inches="tight")
plt.savefig(figpath_png, dpi=300, bbox_inches="tight")
print(f"已写 {figpath_pdf}")
print(f"已写 {figpath_png}")
plt.close()
print("\nDONE.")
