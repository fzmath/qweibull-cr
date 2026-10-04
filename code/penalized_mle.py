# -*- coding: utf-8 -*-
"""审稿人 P0: 对比四种估计对目标模式形状 q_E 的表现。
方法: MLE / Penalized-MLE(仅 q 二次惩罚, sigma_q=0.5, mu=1) / MAP(全后验众数) /
      Bayes posterior median / Bayes posterior mean。
格子: qtrue in {0.8,1.0,1.45} x {strong(rho=.6), weak(rho=4.0)}, R 次重复。
数据生成与 sim_grid.py 完全一致 (seed=2026, 每格独立重置 rng)。
用法: python penalized_mle.py          # R=40 全量
      PEN_R=3 python penalized_mle.py # smoke"""
import sys, os, time, json
import numpy as np
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
from qweibull import Q_MIN, Q_MAX
from bayes_model import _loglik_and_grad_raw
from cr_bayes_model import default_cr_prior, cr_logpost_grad
from nuts import sample_nuts, numerical_hessian
from scipy.optimize import minimize
from sim_grid import median_scale, rnd_qw, cr_ng

ET, BT, BC = 100.0, 2.0, 3.0
n = 200
WU, PD = 200, 300
R = int(os.environ.get("PEN_R", "40"))
SIGMA_Q = 0.5          # 仅惩罚 q 的二次惩罚宽度
OUT = r"G:\OPT\q-Weibull竞争风险论文\data\penalized_mle.json"

GRID = [(0.8, "strong", 0.6), (0.8, "weak", 4.0),
        (1.0, "strong", 0.6), (1.0, "weak", 4.0),
        (1.45, "strong", 0.6), (1.45, "weak", 4.0)]


def cr_ng_pen(theta, t, cause, sigma_q=SIGMA_Q):
    """q-W 竞争风险负对数似然 + 仅对 q 的二次惩罚 (mu=1)。
    空间 theta=[xq1,xl1,xb1,xq2,xl2,xb2], q=2-exp(xq), lam=exp(xl), beta=exp(xb)。"""
    neg = 0.0; g = np.zeros(6)
    for j in range(2):
        xq, xl, xb = theta[3*j:3*j+3]
        q = 2 - np.exp(xq); lam = np.exp(xl); beta = np.exp(xb)
        if q < Q_MIN or q > Q_MAX:
            return 1e10, np.zeros(6)
        ll, gq, gl, gb = _loglik_and_grad_raw(q, lam, beta, t, (cause == j).astype(int))
        if not np.isfinite(ll):
            return 1e10, np.zeros(6)
        neg += ll
        g[3*j]   = gq * -np.exp(xq)
        g[3*j+1] = gl * lam
        g[3*j+2] = gb * beta
        # 惩罚 P = -0.5*((q-1)/sigma_q)^2  (只惩罚 q)
        neg += -0.5 * ((q - 1.0) / sigma_q) ** 2
        g[3*j] += (q - 1.0) / sigma_q**2 * np.exp(xq)   # dP/dxq
    return -neg, -g


def fit_penalized(t, cause, th_mle):
    """多起点 Penalized-MLE, 返回 q_E 估计。"""
    def obj(z):
        return cr_ng_pen(z, t, cause)
    starts = [th_mle, th_mle.copy()]
    # 第二起点: 把两个 q 都压向 1 (xq=0)
    starts[1][0] = 0.0; starts[1][3] = 0.0
    best = None
    for s in starts:
        try:
            r = minimize(obj, s, jac=True, method="BFGS", options=dict(maxiter=500))
            if best is None or r.fun < best.fun:
                best = r
        except Exception:
            pass
    if best is None:
        return np.nan
    return 2 - np.exp(best.x[0])


def fit_map(t, cause, th_mle):
    """MAP = 最大化 cr_logpost (与 NUTS 同一后验), 解析梯度, 从 MLE(换 xe 坐标) 出发。"""
    priors = default_cr_prior(t, cause)
    xe = []
    for j in range(2):
        xq, xl, xb = th_mle[3*j:3*j+3]; ll = np.exp(xl); bb = np.exp(xb)
        xe += [xq, np.log(ll ** (-1 / bb)), xb]
    xe = np.array(xe)

    def npost(x):
        lp, g = cr_logpost_grad(x, t, cause, priors)
        if not np.isfinite(lp):
            return 1e10, np.zeros(6)
        return -lp, -g
    try:
        r = minimize(npost, xe, jac=True, method="BFGS", options=dict(maxiter=500))
        return 2 - np.exp(r.x[0])
    except Exception:
        return np.nan


def one_combo(qtrue, sname, rho):
    rng = np.random.default_rng(2026)
    tmed_t = ET * median_scale(qtrue, BT)
    eta_c = rho * tmed_t / median_scale(1, BC)
    mle, pen, map_ = [], [], []
    bmed, bmean = [], []
    for r in range(R):
        T0 = rnd_qw(qtrue, ET, BT, n, rng)
        T1 = rnd_qw(1.0, eta_c, BC, n, rng)
        t = np.minimum(T0, T1); cause = (T1 < T0).astype(int)
        # --- MLE (与 sim_grid 完全一致) ---
        x0 = np.concatenate([[np.log(2-qtrue), np.log(ET**-BT), np.log(BT)],
                             [np.log(1), np.log(eta_c**-BC), np.log(BC)]])
        th = minimize(lambda z: cr_ng(z, t, cause), x0, jac=True, method="BFGS",
                      options=dict(maxiter=500)).x
        mle.append(2 - np.exp(th[0]))
        # --- Penalized-MLE ---
        pen.append(fit_penalized(t, cause, th))
        # --- MAP ---
        map_.append(fit_map(t, cause, th))
        # --- Bayes (NUTS, 与 sim_grid 一致) ---
        priors = default_cr_prior(t, cause)
        gradf = lambda x: cr_logpost_grad(x, t, cause, priors)
        xe = []
        for j in range(2):
            xq, xl, xb = th[3*j:3*j+3]; ll = np.exp(xl); bb = np.exp(xb)
            xe += [xq, np.log(ll ** (-1 / bb)), xb]
        xe = np.array(xe)
        dg = sample_nuts([xe], gradf, n_draw=WU+PD, warmup=WU, target_accept=0.90,
                         init_hessian=-numerical_hessian(gradf, xe, 1e-4))
        qp = 2 - np.exp(dg["chains"][0, WU:, 0])
        bmed.append(np.median(qp)); bmean.append(qp.mean())

    def summ(arr):
        a = np.asarray(arr, float); a = a[np.isfinite(a)]
        return dict(bias=float((a - qtrue).mean()),
                    rmse=float(np.sqrt(((a - qtrue)**2).mean())),
                    mae=float(np.abs(a - qtrue).mean()),
                    n=int(len(a)))
    return dict(qtrue=qtrue, rho_name=sname, rho=rho, eta_c=float(eta_c), n=n, R=int(R),
                mle=summ(mle), penalized=summ(pen), map=summ(map_),
                bayes_median=summ(bmed), bayes_mean=summ(bmean))


if __name__ == "__main__":
    cells = []
    for (qtrue, sname, rho) in GRID:
        t0 = time.time()
        res = one_combo(qtrue, sname, rho)
        cells.append(res)
        print(f"q={qtrue} {sname}: MLE rmse={res['mle']['rmse']:.3f} "
              f"PEN rmse={res['penalized']['rmse']:.3f} MAP rmse={res['map']['rmse']:.3f} "
              f"Bmed rmse={res['bayes_median']['rmse']:.3f} Bmean rmse={res['bayes_mean']['rmse']:.3f} "
              f"[{(time.time()-t0)/60:.1f}m]", flush=True)
    json.dump(dict(R=int(R), sigma_q=SIGMA_Q, cells=cells),
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("WROTE", OUT)
