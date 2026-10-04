# -*- coding: utf-8 -*-
"""审稿人 P0-3: 两种采样坐标参数化的定量对比。

参数化A (现有):  x_A = [xq, xe, xb],  q=2-exp(xq), eta=exp(xe), beta=exp(xb)
                                          lam=eta^-beta (派生)
参数化B (新):    x_B = [xq, xlam, xb], q=2-exp(xq), lam=exp(xlam), beta=exp(xb)
                  (lambda 直接取对数; eta=lam^(-1/beta) 派生)

B 是 A 的精确坐标重参数化 (同一后验分布, 仅采样坐标不同):
    xlam = -beta*xe  =>  xe = -xlam/beta
    测度 Jacobian: dxe = dxlam/beta  =>  logpost_B = logpost_A - log(beta)
    梯度链式:  gB_xq=gA_xq;  gB_xlam=gA_xe*(-1/beta);
              gB_xb = gA_xb + gA_xe*(xlam/beta) - 1
(已在 _smoke_paramB.py 用有限差分验证, 误差~1e-8)

在 voltage 真实高压数据上各跑 2 条 NUTS 链, 报告:
  - 后验相关矩阵 (原始参数 6 维 [qE,etaE,bE,qD,etaD,bD]) 的 max/mean |off-diag|
  - 众数处负对数后验 Hessian (信息阵) 条件数 kappa=lambda_max/lambda_min
  - ESS, ESS/sec, divergences, mean tree depth, R-hat
输出: data/param_compare.json, figures/fig8_param_corr.pdf/.png
"""
import sys, os, json, time
import numpy as np
import pandas as pd
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
from cr_bayes_model import default_cr_prior, cr_logpost_grad
from nuts import sample_nuts, numerical_hessian
from scipy.optimize import minimize

DATA = r"G:\OPT\q-Weibull竞争风险论文\data\voltage.csv"
OUT_JSON = r"G:\OPT\q-Weibull竞争风险论文\data\param_compare.json"
FIG_PDF = r"G:\OPT\q-Weibull竞争风险论文\figures\fig8_param_corr.pdf"
FIG_PNG = r"G:\OPT\q-Weibull竞争风险论文\figures\fig8_param_corr.png"

# ---------------- 数据 ----------------
df = pd.read_csv(DATA)
t = df["hours"].to_numpy(float)
fm = df["failure_mode"].astype(str).str.lower().to_numpy()
cause = np.where(fm == "e", 0, np.where(fm == "d", 1, -1)).astype(int)
N = len(t); nE = int((cause == 0).sum()); nD = int((cause == 1).sum()); nC = int((cause == -1).sum())
priors = default_cr_prior(t, cause, q_scale=0.5)

# ---------------- 坐标变换 ----------------
def xB_to_xA(xB):
    xA = np.empty_like(xB)
    for j in range(2):
        xq, xlam, xb = xB[3*j:3*j+3]
        xA[3*j:3*j+3] = [xq, -xlam / np.exp(xb), xb]
    return xA

def gradf_A(x):
    return cr_logpost_grad(x, t, cause, priors)

def gradf_B(xB):
    xA = xB_to_xA(xB)
    lpA, gA = cr_logpost_grad(xA, t, cause, priors)
    if not np.isfinite(lpA):
        return -np.inf, np.zeros_like(xB)
    lpB = lpA
    gB = np.zeros_like(xB)
    for j in range(2):
        xq, xlam, xb = xB[3*j:3*j+3]
        beta = np.exp(xb)
        lpB -= np.log(beta)
        gB[3*j]     = gA[3*j]
        gB[3*j + 1] = gA[3*j + 1] * (-1.0 / beta)
        gB[3*j + 2] = gA[3*j + 2] + gA[3*j + 1] * (xlam / beta) - 1.0
    return lpB, gB

def negpost_A(x):
    lp, g = gradf_A(x)
    return (-lp, -g) if np.isfinite(lp) else (1e10, np.zeros_like(x))

def negpost_B(x):
    lp, g = gradf_B(x)
    return (-lp, -g) if np.isfinite(lp) else (1e10, np.zeros_like(x))

# ---------------- 原始参数还原 ----------------
def to_original_A(x):
    out = []
    for j in range(2):
        xq, xe, xb = x[..., 3*j:3*j+3].T
        q = 2.0 - np.exp(xq); eta = np.exp(xe); beta = np.exp(xb)
        out += [q, eta, beta]
    return np.array(out).T   # shape (...,6)

def to_original_B(x):
    out = []
    for j in range(2):
        xq, xlam, xb = x[..., 3*j:3*j+3].T
        q = 2.0 - np.exp(xq); lam = np.exp(xlam); beta = np.exp(xb)
        eta = lam ** (-1.0 / beta)
        out += [q, eta, beta]
    return np.array(out).T

# ---------------- 找众数 ----------------
def find_mode(negpost, x0):
    r = minimize(negpost, x0, jac=True, method="BFGS",
                 options=dict(maxiter=2000, gtol=1e-7))
    return r.x, -r.fun

# 初始猜测 (贴近 voltage 已知估计: etaE~900,betaE~0.68; etaD~345,betaD~5.3; q~1.3)
xq0 = np.log(2 - 1.3)
etaE0, bE0, etaD0, bD0 = 900.0, 0.68, 345.0, 5.3
xA0 = np.array([xq0, np.log(etaE0), np.log(bE0), xq0, np.log(etaD0), np.log(bD0)])
lamE0, lamD0 = etaE0 ** -bE0, etaD0 ** -bD0
xB0 = np.array([xq0, np.log(lamE0), np.log(bE0), xq0, np.log(lamD0), np.log(bD0)])

modeA, lpmodeA = find_mode(negpost_A, xA0)
modeB, lpmodeB = find_mode(negpost_B, xB0)
print("mode A logpost=%.4f" % lpmodeA)
print("mode B logpost=%.4f (应=modeA - sum log beta; 坐标不同不直接相等)" % lpmodeB)

# ---------------- 条件数: 众数处负对数后验 Hessian (信息阵) ----------------
def condnum(gradf, mode):
    Hlog = numerical_hessian(gradf, mode, epsilon=1e-4)   # Hessian of logpost
    H = -Hlog                                            # Hessian of neg-logpost = info
    H = 0.5 * (H + H.T)
    w = np.linalg.eigvalsh(H)
    wpos = w[w > 0]
    kappa = float(wpos.max() / wpos.min())
    return kappa, w

kappaA, eigA = condnum(gradf_A, modeA)
kappaB, eigB = condnum(gradf_B, modeB)
print("kappa A=%.3e  B=%.3e" % (kappaA, kappaB))

# ---------------- NUTS 采样 ----------------
def run_case(name, gradf, mode, to_original, seed0):
    # 两条链, 众数附近小幅扰动
    rng = np.random.default_rng(seed0)
    inits = [mode + 0.05 * rng.standard_normal(6) for _ in range(2)]
    t0 = time.time()
    dg = sample_nuts(inits, gradf, n_draw=400 + 600, warmup=400,
                     target_accept=0.90,
                     init_hessian=-numerical_hessian(gradf, mode, 1e-4),
                     adapt_mass=True)
    wall = time.time() - t0
    # 后验样本 (采样坐标) -> 原始参数
    post_samp = dg["chains"][:, 400:, :]              # (2,600,6)
    orig = to_original(post_samp.reshape(-1, 6)).reshape(2, 600, 6)
    Cmat = np.corrcoef(orig.reshape(-1, 6).T)
    off = Cmat.copy()
    np.fill_diagonal(off, 0.0)
    max_off = float(np.max(np.abs(off)))
    mean_off = float(np.mean(np.abs(off)))
    res = dict(
        wall_time=float(wall),
        ess=[float(x) for x in dg["ess"]],
        ess_total=float(np.sum(dg["ess"])),
        ess_per_sec=float(dg["ess_per_sec"]),
        divergent=int(dg["divergent"]),
        divergent_warmup=int(dg["divergent_warmup"]),
        mean_tree_depth=float(dg["depth"]),
        rhat=[float(x) for x in dg["rhat"]],
        accept=float(dg["accept"]),
        postcorr_max_offdiag=max_off,
        postcorr_mean_offdiag=mean_off,
        postcorr=Cmat.tolist(),
        mode=[float(x) for x in mode],
    )
    print("[%s] div=%d ess_sec=%.1f depth=%.2f max|offcorr|=%.3f mean|offcorr|=%.3f rhat_max=%.3f"
          % (name, res["divergent"], res["ess_per_sec"], res["mean_tree_depth"],
             max_off, mean_off, np.nanmax(dg["rhat"])))
    return res, Cmat

resA, corrA = run_case("A (q,logeta,logb)", gradf_A, modeA, to_original_A, 11)
resB, corrB = run_case("B (q,loglam,logb)", gradf_B, modeB, to_original_B, 22)

# ---------------- 结论 ----------------
ratio_kappa = kappaB / kappaA
ess_ratio = resB["ess_per_sec"] / resA["ess_per_sec"]
off_ratio = resB["postcorr_max_offdiag"] / max(resA["postcorr_max_offdiag"], 1e-9)
if (ratio_kappa > 0.1) and (ess_ratio < 1.5) and (off_ratio > 0.8):
    verdict = ("没有证据表明 (q,loglambda,logbeta) 比 (q,logeta,logbeta) 更正交; "
               "应把论文中 'approximately orthogonal coordinates' 措辞降级为 "
               "'a more weakly coupled parameterization'。")
else:
    verdict = ("(q,loglambda,logbeta) 确实显著改善了坐标正交性/采样效率, "
               "可保留原措辞。")

out = dict(
    dataset="voltage", n=N, E=nE, D=nD, censored=nC,
    setup=dict(warmup=400, post=600, chains=2, target_accept=0.90,
               note="B=A的精确坐标重参数化(含测度Jacobian -log beta); 梯度经有限差分验证"),
    A=dict(kappa=kappaA, eigmin=float(eigA.min()), eigmax=float(eigA.max()), **resA),
    B=dict(kappa=kappaB, eigmin=float(eigB.min()), eigmax=float(eigB.max()), **resB),
    ratio_kappa_B_over_A=float(ratio_kappa),
    ratio_esspersec_B_over_A=float(ess_ratio),
    verdict=verdict,
)
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("\nsaved:", OUT_JSON)
print("VERDICT:", verdict)

# ---------------- 图: 后验相关热图 ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
labels = ["q_E", "eta_E", "beta_E", "q_D", "eta_D", "beta_D"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, C, title in zip(axes, [corrA, corrB],
                        ["A: (q, log eta, log beta)", "B: (q, log lam, log beta)"]):
    im = ax.imshow(C, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(6)); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(6)); ax.set_yticklabels(labels)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, "%.2f" % C[i, j], ha="center", va="center",
                    fontsize=7, color="black")
    ax.set_title(title)
fig.colorbar(im, ax=axes, fraction=0.025, pad=0.04)
plt.savefig(FIG_PDF, bbox_inches="tight")
plt.savefig(FIG_PNG, bbox_inches="tight", dpi=150)
print("saved:", FIG_PDF, FIG_PNG)
