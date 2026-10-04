# -*- coding: utf-8 -*-
"""加分: 多采样器 benchmark (q=1.45 strong, 6 维)。
RW-MH(预条件,给后验协方差) / Adaptive Metropolis(Haario) / 预条件 HMC(固定L) / NUTS。
报 ESS/s, ESS/梯度评估(HMC系), 墙钟, 接受率, 发散。"""
import sys, time, json
import numpy as np
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
from qweibull import Q_MIN, Q_MAX
from bayes_model import _loglik_and_grad_raw
from cr_bayes_model import default_cr_prior, cr_logpost_grad
from nuts import sample_nuts, numerical_hessian, mass_from_hessian, _ess
from sim_grid import rnd_qw
from scipy.optimize import minimize


def cr_mle_neggrad(theta, t, cause):
    neg = 0.0; g = np.zeros(6)
    for j in range(2):
        xq, xl, xb = theta[3*j:3*j+3]
        q = 2-np.exp(xq); lam = np.exp(xl); beta = np.exp(xb)
        if q < Q_MIN or q > Q_MAX:
            return 1e10, np.zeros(6)
        ll, gq, gl, gb = _loglik_and_grad_raw(q, lam, beta, t,
                                              (cause == j).astype(int))
        if not np.isfinite(ll):
            return 1e10, np.zeros(6)
        neg += ll
        g[3*j] = gq*-np.exp(xq); g[3*j+1] = gl*lam; g[3*j+2] = gb*beta
    return -neg, -g


def sample_mh(x0, gradf, n, warmup, Lprop, rng, target=0.234):
    x = x0.copy(); lp, _ = gradf(x); out = []; acc = 0
    sc = 2.4/np.sqrt(len(x0))
    for i in range(n):
        xn = x+sc*(Lprop@rng.standard_normal(len(x0)))
        lpn, _ = gradf(xn)
        if np.log(rng.random()) < lpn-lp:
            x, lp = xn, lpn; acc += 1
        if i < warmup and i % 40 == 39:
            sc *= np.exp((acc/40-target)*1.5)
            sc = np.clip(sc, 1e-3, 10.0); acc = 0
        if i >= warmup:
            out.append(x)
    return np.array(out), acc/max(n-warmup, 1), sc


def sample_am(x0, gradf, n, wu, rng, target=0.234):
    x = x0.copy(); lp, _ = gradf(x); d = len(x)
    C = np.eye(d); sc = 2.4/np.sqrt(d); acc = 0; out = []; hist = []
    for i in range(n):
        if i < wu:
            hist.append(x.copy())
            if i == 100 or (i > 100 and i % 100 == 0):
                S = np.cov(np.array(hist).T)
                C = S + 1e-8*np.eye(d)
        try:
            Lc = np.linalg.cholesky(C + 1e-10*np.eye(d))
        except Exception:
            Lc = np.linalg.cholesky(np.eye(d))
        xn = x + sc*(Lc@rng.standard_normal(d))
        lpn, _ = gradf(xn)
        if np.log(rng.random()) < lpn-lp:
            x, lp = xn, lpn; acc += 1
        if i < wu and i % 40 == 39:
            sc *= np.exp((acc/40-target)*1.5)
            sc = np.clip(sc, 1e-3, 10); acc = 0
        if i >= wu:
            out.append(x.copy())
    return np.array(out), acc/max(n-wu, 1)


def sample_hmc_precond(x0, gradf, n, wu, Sinv, rng, L=20, target=0.9):
    x = x0.copy(); lp, g = gradf(x); d = len(x); eps = 0.05
    acc = 0; out = []; ngrad = 0
    for i in range(n):
        r = rng.standard_normal(d)
        H0 = lp-0.5*r@(Sinv@r)
        xn, rn, gn, lpn = x.copy(), r.copy(), g, lp
        rn = rn+0.5*eps*gn
        for k in range(L):
            xn = xn+eps*(Sinv@rn)
            lpn, gn = gradf(xn); ngrad += 1
            if k < L-1:
                rn = rn+eps*gn
        rn = rn+0.5*eps*gn
        H1 = lpn-0.5*rn@(Sinv@rn)
        if np.log(rng.random()) < H0-H1:
            x, g, lp = xn, gn, lpn; acc += 1
        if i < wu and i % 20 == 19:
            eps *= np.exp((acc/20-target)*0.7)
            eps = np.clip(eps, 1e-4, 2.0); acc = 0
        if i >= wu:
            out.append(x.copy())
    return np.array(out), acc/max(n-wu, 1), ngrad


# ---- 数据与起点 (q1.45 strong, ET76.7, eta_c68) ----
qtrue, eta2 = 1.45, 68.0
rng = np.random.default_rng(11)
T0 = rnd_qw(qtrue, 76.7, 2, 200, rng); T1 = rnd_qw(1, eta2, 3, 200, rng)
t = np.minimum(T0, T1); cause = (T1 < T0).astype(int)
priors = default_cr_prior(t, cause)
gradf_real = lambda x: cr_logpost_grad(x, t, cause, priors)
x0lam = np.concatenate([[np.log(1), np.log(76.7**-2), np.log(2)],
                        [np.log(1), np.log(eta2**-3), np.log(3)]])
th = minimize(lambda z: cr_mle_neggrad(z, t, cause), x0lam, jac=True,
              method="BFGS").x
xe = []
for j in range(2):
    xq, xl, xb = th[3*j:3*j+3]; ll = np.exp(xl); bb = np.exp(xb)
    xe += [xq, np.log(ll**(-1/bb)), xb]
xe = np.array(xe)
H0 = -numerical_hessian(gradf_real, xe, 1e-4)
Mmat, Cov, _LcM = mass_from_hessian(H0)   # Mmat=精度, Cov=M^-1=后验协方差
Sinv = Cov                                # HMC 位置更新 x+eps M^-1 r
Lprop = np.linalg.cholesky(Cov+1e-8*np.eye(6))  # MH 提议协方差=后验协方差(正确)

results = {}

# RW-MH
t0 = time.time(); mh, amh, _ = sample_mh(xe, gradf_real, 6000, 2000, Lprop,
                                         np.random.default_rng(2)); sec = time.time()-t0
ess = _ess(mh[None]).sum()
results["RW-MH"] = dict(ess=float(ess), sec=float(sec), ess_sec=float(ess/sec),
                        accept=float(amh), n_grad=0, ess_grad=None)

# Adaptive Metropolis
t0 = time.time(); am, aam = sample_am(xe, gradf_real, 6000, 2000,
                                      np.random.default_rng(2)); sec = time.time()-t0
ess = _ess(am[None]).sum()
results["Adaptive-MH"] = dict(ess=float(ess), sec=float(sec), ess_sec=float(ess/sec),
                              accept=float(aam), n_grad=0, ess_grad=None)

# 预条件 HMC
t0 = time.time(); hc, ahc, ng = sample_hmc_precond(xe, gradf_real, 1500, 500, Sinv,
                                                   np.random.default_rng(2)); sec = time.time()-t0
ess = _ess(hc[None]).sum()
results["HMC(precond)"] = dict(ess=float(ess), sec=float(sec), ess_sec=float(ess/sec),
                               accept=float(ahc), n_grad=int(ng),
                               ess_grad=float(ess/max(ng, 1)))

# NUTS
t0 = time.time(); dN = sample_nuts([xe], gradf_real, n_draw=1000, warmup=500,
                                   target_accept=0.9, init_hessian=H0); sec = time.time()-t0
ess = dN["ess"].sum(); ng = int(dN.get("n_grad", 0))
results["NUTS"] = dict(ess=float(ess), sec=float(sec), ess_sec=float(ess/sec),
                       accept=float(dN["accept"]), n_grad=ng,
                       ess_grad=(float(ess/ng) if ng else None),
                       divergent=int(dN["divergent"]))

for nm, r in results.items():
    eg = f"{r['ess_grad']:.3f}" if r['ess_grad'] is not None else "NA"
    dv = f" div={r['divergent']}" if 'divergent' in r else ""
    print(f"{nm:15} ESS/s={r['ess_sec']:6.1f} ESS/grad={eg:>6} "
          f"acc={r['accept']:.2f} ESS={r['ess']:.0f} sec={r['sec']:.1f}{dv}")

json.dump(results, open(r"G:\OPT\q-Weibull竞争风险论文\data\benchmark_results.json",
                        "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("saved benchmark_results.json")
