# -*- coding: utf-8 -*-
"""必改3: voltage Bayes Weibull 竞争风险 (q 固定1, 4维) NUTS, 与 q-W Bayes 对比。"""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
from nuts import sample_nuts, numerical_hessian
from cr_bayes_model import default_cr_prior
from bayes_model import _loglik_and_grad_raw
from sim_grid import cr_ng_w
from scipy.optimize import minimize

df = pd.read_csv(r"G:\OPT\q-Weibull竞争风险论文\data\voltage.csv")
t = df["hours"].to_numpy(float)
fm = df["failure_mode"].to_numpy(str)
cause = np.full(len(df), -1)
cause[fm == "E"] = 0
cause[fm == "D"] = 1
priors = default_cr_prior(t, cause)


def wcr_logpost_grad(x):
    total = 0.0
    g = np.zeros(4)
    for j in range(2):
        xe, xb = x[2*j], x[2*j+1]
        eta = np.exp(xe); beta = np.exp(xb); lam = eta**(-beta)
        status_j = (cause == j).astype(int)
        ll, gq, gl, gb = _loglik_and_grad_raw(1.0, lam, beta, t, status_j)
        pr = priors[j]
        lpe = -0.5*((xe-pr.mu_e)/pr.sigma_e)**2
        lpb = -0.5*((xb-pr.mu_b)/pr.sigma_b)**2
        total += ll + lpe + lpb
        g[2*j] = -beta*lam*gl - (xe-pr.mu_e)/pr.sigma_e**2
        g[2*j+1] = beta*(gb-gl*lam*xe) - (xb-pr.mu_b)/pr.sigma_b**2
    if not np.isfinite(total):
        return -np.inf, np.zeros(4)
    return total, g


# MLE Weibull CR 初始 (4维, xl=log lam, xb)
init = []
for j in range(2):
    pr = priors[j]
    xl0 = -1.5*pr.mu_e
    init += [xl0, np.log(1.5)]
thw = minimize(lambda z: cr_ng_w(z, t, cause), np.array(init), jac=True,
               method="BFGS", options=dict(maxiter=500)).x
x_init = np.array([-thw[0]/np.exp(thw[1]), thw[1],
                   -thw[2]/np.exp(thw[3]), thw[3]])
print("MLE W CR loglik 对应 neg=", cr_ng_w(thw, t, cause)[0])

gradf = wcr_logpost_grad
H0 = -numerical_hessian(gradf, x_init, epsilon=1e-4)
diag = sample_nuts([x_init], gradf, n_draw=2000, warmup=1000,
                   target_accept=0.9, init_hessian=H0)
post = diag["chains"][0, 1000:, :]      # N,4 = [xe0,xb0,xe1,xb1]

rows = {}
print("\n参数      后验均值  中位数    2.5%     97.5%")
for j, lab in [(0, "E"), (1, "D")]:
    xe, xb = post[:, 2*j], post[:, 2*j+1]
    eta, beta = np.exp(xe), np.exp(xb)
    for nm, c in [(f"eta_{lab}", eta), (f"beta_{lab}", beta)]:
        lo, hi = np.percentile(c, [2.5, 97.5])
        rows[nm] = dict(mean=float(c.mean()), median=float(np.median(c)),
                        lo=float(lo), hi=float(hi))
        print(f"{nm:<9} {c.mean():8.2f} {np.median(c):8.2f} {lo:8.2f} {hi:8.2f}")

# 逐 draw loglik (Weibull CR)
def w_loglik(x):
    tot = 0.0
    for j in range(2):
        xe, xb = x[2*j], x[2*j+1]
        eta = np.exp(xe); beta = np.exp(xb); lam = eta**(-beta)
        tot += _loglik_and_grad_raw(1.0, lam, beta, t,
                                    (cause == j).astype(int))[0]
    return tot
ll_draw = np.array([w_loglik(x) for x in post])
print(f"\n后验 loglik (W CR): mean={ll_draw.mean():.2f} median={np.median(ll_draw):.2f}")

print("\nRhat =", diag["rhat"], " ESS =", diag["ess"].astype(int))
print("depth =", round(diag["depth"], 2), " accept =", round(diag["accept"], 3),
      " post发散 =", diag["divergent"], " warmup =", diag["divergent_warmup"],
      " t =", round(diag["elapsed"], 1), "s")

json.dump(dict(rows=rows, post_loglik_mean=float(ll_draw.mean()),
               post_loglik_median=float(np.median(ll_draw)),
               rhat=[float(v) for v in diag["rhat"]],
               ess=[float(v) for v in diag["ess"]], divergent=int(diag["divergent"]),
               elapsed=float(diag["elapsed"])),
          open(r"G:\OPT\q-Weibull竞争风险论文\data\bayes_weibull_cr.json", "w",
               encoding="utf-8"), ensure_ascii=False, indent=2)
np.save(r"G:\OPT\q-Weibull竞争风险论文\data\post_chains_weibull.npy", diag["chains"])
print("已保存 bayes_weibull_cr.json 与 post_chains_weibull.npy")
