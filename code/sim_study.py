# -*- coding: utf-8 -*-
"""仿真: 竞争截断下 q 的 MLE vs Bayes (偏差/RMSE/95覆盖率)。
目标模式(0): q_true, beta=2, eta=100; 竞争模式(1): Weibull, beta=2, eta2。"""
import sys, time, json
import numpy as np
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
from qweibull import Q_MIN, Q_MAX
from bayes_model import _loglik_and_grad_raw
from cr_bayes_model import default_cr_prior, cr_logpost_grad
from nuts import sample_nuts, numerical_hessian
from scipy.optimize import minimize

def rnd_qw(q, eta, beta, n, rng):
    u = np.maximum(rng.random(n), 1e-12)
    lam = eta**(-beta)
    if abs(q-1) < 1e-9:
        return eta*(-np.log(u))**(1/beta)
    d = 1-q; p = (2-q)/d
    inner = u**(1/p)
    return ((1-inner)/(d*lam))**(1/beta)

def cr_mle_neggrad(theta, t, cause):
    neg = 0.0; g = np.zeros(6)
    for j in range(2):
        xq, xl, xb = theta[3*j:3*j+3]
        q = 2-np.exp(xq); lam = np.exp(xl); beta = np.exp(xb)
        if q < Q_MIN or q > Q_MAX:
            return 1e10, np.zeros(6)
        st = (cause == j).astype(int)
        ll, gq, gl, gb = _loglik_and_grad_raw(q, lam, beta, t, st)
        if not np.isfinite(ll):
            return 1e10, np.zeros(6)
        neg += ll
        g[3*j] = gq*(-np.exp(xq)); g[3*j+1] = gl*lam; g[3*j+2] = gb*beta
    return -neg, -g

def fit_mle(t, cause, x_start):
    r = minimize(lambda z: cr_mle_neggrad(z, t, cause), x_start, jac=True,
                 method="BFGS", options=dict(maxiter=500))
    return r.x

scenarios = [
    ("有界-强截断", 0.8, 40.0),
    ("有界-弱截断", 0.8, 400.0),
    ("重尾-强截断", 1.3, 40.0),
    ("重尾-弱截断", 1.3, 400.0),
]
n = 200
R = int(sys.argv[1]) if len(sys.argv) > 1 else 5
WU, PD = 200, 300

results = {}
for name, qtrue, eta2 in scenarios:
    rng = np.random.default_rng(2026)
    mle_q, bay_q, mcov, bcov, sec = [], [], [], [], 0
    for r in range(R):
        T0d = rnd_qw(qtrue, 100.0, 2.0, n, rng)
        T1d = rnd_qw(1.0, eta2, 2.0, n, rng)
        t = np.minimum(T0d, T1d); cause = (T1d < T0d).astype(int)
        x0lam = np.concatenate([[np.log(1.0), np.log(100.0**-2), np.log(2.0)],
                                [np.log(1.0), np.log(eta2**-2), np.log(2.0)]])
        th = fit_mle(t, cause, x0lam)
        q_mle = 2-np.exp(th[0]); mle_q.append(q_mle)
        # MLE 渐近区间 (观测信息)
        def fval(z):
            v, _ = cr_mle_neggrad(z, t, cause); return v, np.zeros(6)
        H = numerical_hessian(fval, th, 1e-3)
        try:
            Ci = np.linalg.inv(H + 1e-6*np.eye(6))
            se = np.exp(th[0])*np.sqrt(max(Ci[0, 0], 0))
            se = min(se, 5.0)
            mcov.append(q_mle-1.96*se <= qtrue <= q_mle+1.96*se)
        except Exception:
            mcov.append(False)
        # Bayes
        priors = default_cr_prior(t, cause)
        def gradf(x):
            return cr_logpost_grad(x, t, cause, priors)
        xe = []
        for j in range(2):
            xq,xl,xb = th[3*j:3*j+3]
            qq=2-np.exp(xq); ll=np.exp(xl); bb=np.exp(xb)
            xe += [xq, np.log(ll**(-1/bb)), xb]
        xe = np.array(xe)
        diag = sample_nuts([xe], gradf, n_draw=WU+PD, warmup=WU, target_accept=0.85,
                           init_hessian=-numerical_hessian(gradf, xe, 1e-4))
        qp = 2-np.exp(diag["chains"][0, WU:, 0])
        bay_q.append(qp.mean())
        lo, hi = np.percentile(qp, [2.5, 97.5])
        bcov.append(lo <= qtrue <= hi)
        sec += diag["elapsed"]
    mle_q, bay_q = np.array(mle_q), np.array(bay_q)
    results[name] = dict(
        qtrue=qtrue,
        mle_bias=float((mle_q-qtrue).mean()),
        mle_rmse=float(np.sqrt(((mle_q-qtrue)**2).mean())),
        mle_cover=float(np.mean(mcov)),
        bay_bias=float((bay_q-qtrue).mean()),
        bay_rmse=float(np.sqrt(((bay_q-qtrue)**2).mean())),
        bay_cover=float(np.mean(bcov)),
        sec=float(sec/R))
    s = results[name]
    print(f"{name} q={qtrue}: MLE bias={s['mle_bias']:+.3f} RMSE={s['mle_rmse']:.3f} "
          f"覆盖={s['mle_cover']:.2f} | Bayes bias={s['bay_bias']:+.3f} "
          f"RMSE={s['bay_rmse']:.3f} 覆盖={s['bay_cover']:.2f} ({s['sec']:.1f}s/rep)")

if R >= 20:
    with open(r"G:\OPT\q-Weibull竞争风险论文\data\sim_results.json", "w",
              encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("已保存 sim_results.json")
