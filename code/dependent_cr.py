# -*- coding: utf-8 -*-
"""P1-10 相依竞争风险小仿真: 独立假设误设对 q_E 估计的影响。

用 shared frailty (Gamma) 引入两模式间相依,
然后用错误假设为独立的 q-W 竞争风险 Bayes NUTS 拟合,
报告 q_E 的 median estimate / bias / RMSE / coverage。

三档相依: (0) 独立; (1) 轻度 ν=5; (2) 较强 ν=2。
R=20, n=200, qtrue=1.45, eta_E=100, beta_E=2, beta_D=3, rho=0.6。
"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
from qweibull import Q_MIN, Q_MAX
from bayes_model import _loglik_and_grad_raw
from cr_bayes_model import default_cr_prior, cr_logpost_grad
from nuts import sample_nuts, numerical_hessian
from scipy.optimize import minimize

# ---- 仿真参数 (照 sim_grid.py) ----
QTRUE = 1.45
ET, BT = 100.0, 2.0       # 目标 q-W 尺度/形状 (E mode)
BC = 3.0                   # 竞争 Weibull 形状 (D mode)
RHO = 0.6                  # strong 档: 竞争模式中位寿命 = 0.6 * 目标中位寿命
n = 200
R = int(os.environ.get("DEP_R", "20"))
WU, PD = 200, 300          # warmup / post-warmup draws (照 sim_grid)

# 三档相依: shared frailty Gamma(shape=nu, scale=1/nu), E[Z]=1, Var(Z)=1/nu
# nu 越大 -> Var 越小 -> 相依越弱
FRAILTY_LEVELS = [
    ("independent", None),   # 无 frailty, Z=1
    ("mild", 5.0),           # 轻度相依
    ("strong", 2.0),         # 较强相依
]

OUTJSON = r"G:\OPT\q-Weibull竞争风险论文\data\dependent_cr.json"


def median_scale(q, beta):
    if abs(q - 1) < 1e-9:
        return np.log(2) ** (1 / beta)
    return ((1 - 0.5 ** ((1 - q) / (2 - q))) / (1 - q)) ** (1 / beta)


def rnd_qw(q, eta, beta, m, rng, u_override=None):
    """q-Weibull 逆 CDF 采样。u_override: 外部给定的 Uniform(0,1) (用于 frailty 变换)。"""
    if u_override is None:
        u = np.maximum(rng.random(m), 1e-12)
    else:
        u = np.maximum(u_override, 1e-12)
    lam = eta ** (-beta)
    if abs(q - 1.0) < 1e-9:
        return eta * (-np.log(u)) ** (1.0 / beta)
    d = 1.0 - q
    return ((1.0 - u ** (1.0 / ((2.0 - q) / d))) / (d * lam)) ** (1.0 / beta)


def cr_ng(theta, t, cause):
    """q-W CR MLE 目标函数 (neg loglik + grad)。"""
    neg = 0.0
    g = np.zeros(6)
    for j in range(2):
        xq, xl, xb = theta[3*j:3*j+3]
        q = 2 - np.exp(xq)
        lam = np.exp(xl)
        beta = np.exp(xb)
        if q < Q_MIN or q > Q_MAX:
            return 1e10, np.zeros(6)
        ll, gq, gl, gb = _loglik_and_grad_raw(q, lam, beta, t, (cause == j).astype(int))
        if not np.isfinite(ll):
            return 1e10, np.zeros(6)
        neg += ll
        g[3*j] = gq * (-np.exp(xq))
        g[3*j+1] = gl * lam
        g[3*j+2] = gb * beta
    return -neg, -g


def gen_dataset(n, qtrue, eta_t, beta_t, eta_c, beta_c, nu, rng):
    """生成竞争风险数据, 可选 shared frailty。

    nu=None: 独立。
    nu=float: Z ~ Gamma(shape=nu, scale=1/nu), 两模式共享。
              R(t|Z) = R_base(t)^Z, 故逆 CDF 用 u_base = u^(1/Z)。
    """
    if nu is None:
        # 独立: 直接采样
        T0 = rnd_qw(qtrue, eta_t, beta_t, n, rng)
        T1 = rnd_qw(1.0, eta_c, beta_c, n, rng)
    else:
        # Shared frailty: Z ~ Gamma(nu, 1/nu)
        Z = rng.gamma(shape=nu, scale=1.0/nu, size=n)
        # 两模式各自的 Uniform
        u0 = rng.random(n)
        u1 = rng.random(n)
        # R(t|Z) = R_base(t)^Z = u  =>  R_base(t) = u^(1/Z)
        u0_base = u0 ** (1.0 / Z)
        u1_base = u1 ** (1.0 / Z)
        T0 = rnd_qw(qtrue, eta_t, beta_t, n, rng, u_override=u0_base)
        T1 = rnd_qw(1.0, eta_c, beta_c, n, rng, u_override=u1_base)

    t = np.minimum(T0, T1)
    cause = (T1 < T0).astype(int)  # 0=E(T0 wins), 1=D(T1 wins)
    return t, cause


def fit_one(t, cause, rng_seed=0):
    """用错误假设为独立的 q-W CR Bayes NUTS 拟合, 返回 q_E 后验中位数和 95% CI。"""
    # MLE init
    x0 = np.concatenate([
        [np.log(2 - QTRUE), np.log(ET ** (-BT)), np.log(BT)],
        [np.log(1), np.log(eta_c ** (-BC)), np.log(BC)]
    ])
    th = minimize(lambda z: cr_ng(z, t, cause), x0, jac=True, method="BFGS",
                  options=dict(maxiter=500)).x

    # Bayes NUTS
    priors = default_cr_prior(t, cause)
    gradf = lambda x: cr_logpost_grad(x, t, cause, priors)
    xe = []
    for j in range(2):
        xq, xl, xb = th[3*j:3*j+3]
        ll = np.exp(xl); bb = np.exp(xb)
        xe += [xq, np.log(ll ** (-1.0 / bb)), xb]
    xe = np.array(xe)
    dg = sample_nuts([xe], gradf, n_draw=WU+PD, warmup=WU, target_accept=0.90,
                      init_hessian=-numerical_hessian(gradf, xe, 1e-4))
    q_post = 2.0 - np.exp(dg["chains"][0, WU:, 0])
    med = float(np.median(q_post))
    lo, hi = np.percentile(q_post, [2.5, 97.5])
    return med, float(lo), float(hi), float(q_post.std(ddof=1)), int(dg["ess"][0]), int(dg["divergent"])


# ---- 计算 eta_c ----
tmed_t = ET * median_scale(QTRUE, BT)
eta_c = RHO * tmed_t / median_scale(1.0, BC)
print(f"目标中位寿命 t_med(E) = {tmed_t:.2f}h")
print(f"竞争 D 中位寿命 = {RHO*tmed_t:.2f}h (rho={RHO}), eta_c={eta_c:.4f}, beta_c={BC}")
print(f"n={n}, R={R}, WU={WU}, PD={PD}")
print(f"frailty levels: {[(name, nu) for name, nu in FRAILTY_LEVELS]}")

# ---- 逐档仿真 ----
all_results = {}
for level_name, nu in FRAILTY_LEVELS:
    print(f"\n===== {level_name} (nu={nu}) =====")
    rng = np.random.default_rng(2026)
    meds = []
    los = []
    his = []
    sd_list = []
    ess_list = []
    div_list = []
    t0 = time.time()
    for r in range(R):
        t, cause = gen_dataset(n, QTRUE, ET, BT, eta_c, BC, nu, rng)
        nE = (cause == 0).sum()
        nD = (cause == 1).sum()
        med, lo, hi, sd, ess, div = fit_one(t, cause)
        meds.append(med); los.append(lo); his.append(hi)
        sd_list.append(sd); ess_list.append(ess); div_list.append(div)
        print(f"  rep {r+1}/{R}: q_E med={med:.3f} [{lo:.3f},{hi:.3f}] "
              f"(nE={nE}, nD={nD}, ess={ess}, div={div})", flush=True)

    meds = np.array(meds); los = np.array(los); his = np.array(his)
    bias = float((meds - QTRUE).mean())
    rmse = float(np.sqrt(((meds - QTRUE) ** 2).mean()))
    mae = float(np.abs(meds - QTRUE).mean())
    coverage = float(np.mean([l <= QTRUE <= h for l, h in zip(los, his)]))
    width = float(np.mean(his - los))
    elapsed = time.time() - t0
    print(f"  -> bias={bias:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, "
          f"coverage={coverage:.3f}, width={width:.3f}, time={elapsed:.0f}s")

    all_results[level_name] = {
        "nu": nu,
        "qtrue": QTRUE,
        "n": n,
        "R": R,
        "qE_median_mean": float(meds.mean()),
        "qE_median_median": float(np.median(meds)),
        "bias": bias,
        "rmse": rmse,
        "mae": mae,
        "coverage": coverage,
        "interval_width": width,
        "post_sd_mean": float(np.mean(sd_list)),
        "ess_mean": float(np.mean(ess_list)),
        "divergences_mean": float(np.mean(div_list)),
        "time_sec": elapsed,
    }

# ---- 写 JSON ----
output = {
    "description": "Shared frailty simulation: effect of dependence misspecification on q_E estimate",
    "method": "shared Gamma frailty: Z~Gamma(nu, 1/nu), R(t|Z)=R_base(t)^Z; fit with WRONG independent q-W CR Bayes NUTS",
    "qtrue": QTRUE,
    "eta_E": ET,
    "beta_E": BT,
    "beta_D": BC,
    "rho": RHO,
    "eta_D": float(eta_c),
    "n_per_rep": n,
    "R": R,
    "warmup": WU,
    "post_draw": PD,
    "levels": all_results,
}
with open(OUTJSON, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"\n已写 {OUTJSON}")
print("\n===== SUMMARY: q_E estimate =====")
print(f"{'Level':<15} {'bias':>8} {'RMSE':>8} {'coverage':>10} {'width':>8}")
for name, nu in FRAILTY_LEVELS:
    r = all_results[name]
    print(f"{name:<15} {r['bias']:>8.4f} {r['rmse']:>8.4f} {r['coverage']:>10.3f} {r['interval_width']:>8.3f}")
print("\nDONE.")
