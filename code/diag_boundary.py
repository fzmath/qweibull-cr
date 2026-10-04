# -*- coding: utf-8 -*-
"""诊断: voltage E 模式沿 q_E 的 profile 似然 (固定 D 在 MLE), 看 q_E->2 边界
是真实高似然还是数值外推伪影。"""
import numpy as np, pandas as pd
from bayes_model import _loglik_and_grad_raw
from scipy.optimize import minimize

df = pd.read_csv(r"G:\OPT\q-Weibull竞争风险论文\data\voltage.csv")
t = df["hours"].to_numpy(float)
fm = df["failure_mode"].to_numpy(str)
cause = np.full(len(df), -1); cause[fm == "E"] = 0; cause[fm == "D"] = 1
stE = (cause == 0).astype(int); stD = (cause == 1).astype(int)

# D 固定 MLE
qD, lamD, bD = 0.9345138, 1.5741296e-14, 5.4219912
llD = _loglik_and_grad_raw(qD, lamD, bD, t, stD)[0]


def e_ng(th, q):
    lam, beta = np.exp(th[0]), np.exp(th[1])
    ll, gq, gl, gb = _loglik_and_grad_raw(q, lam, beta, t, stE)
    if not np.isfinite(ll):
        return 1e12, np.zeros(2)
    return -ll, -np.array([gl*lam, gb*beta])


print("q_E    E logL (profile)   total(+D)    eta_E      beta_E")
for q in [0.8, 1.0, 1.2, 1.24, 1.4, 1.6, 1.7, 1.8, 1.9, 1.93]:
    best = None
    # 多起点 (正常 eta~600, 及极端)
    for eta0 in [600.0, 50.0, 3000.0]:
        x0 = np.array([np.log(eta0**-0.7), np.log(0.7)])
        r = minimize(lambda z: e_ng(z, q), x0, jac=True, method="BFGS",
                     options=dict(maxiter=800))
        if best is None or r.fun < best.fun:
            best = r
    lam = np.exp(best.x[0]); beta = np.exp(best.x[1]); eta = lam**(-1/beta)
    llE = -best.fun
    print(f"{q:<5} {llE:12.3f} {llE+llD:13.3f} {eta:10.2f} {beta:8.3f}")
