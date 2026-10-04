# -*- coding: utf-8 -*-
"""D 模式 beta 剖面似然: 固定 beta_D, 优化其余3参数。"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from cr_bayes_model import default_cr_prior
from bayes_model import _loglik_and_grad_raw

df = pd.read_csv(r"G:\OPT\q-Weibull竞争风险论文\data\voltage.csv")
t = df["hours"].to_numpy(float)
fm = df["failure_mode"].to_numpy(str)
cause = np.full(len(df), -1)
cause[fm == "E"] = 0; cause[fm == "D"] = 1

def neg(y, beta_D):
    # y=[loglam_E,logbeta_E,loglam_D]
    x = np.array([y[0], y[1], y[2], np.log(beta_D)])
    total = 0.0
    for j in range(2):
        xl, xb = x[2*j], x[2*j+1]
        lam, beta = np.exp(xl), np.exp(xb)
        st = (cause == j).astype(int)
        ll, _, _, _ = _loglik_and_grad_raw(1.0, lam, beta, t, st)
        total += ll
    return -total

y0 = np.array([np.log(0.0112337), np.log(0.63537), np.log(6.1394e-15)])
print("beta_D   profile logL")
for bd in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.6, 6.0, 6.5, 7.0]:
    best = np.inf
    for sc in [y0, y0+np.array([0,0,3.0]), y0+np.array([2,0.5,5])]:
        r = minimize(neg, sc, args=(bd,), method="Nelder-Mead",
                     options=dict(maxiter=4000, xatol=1e-5))
        best = min(best, r.fun)
    print(f"{bd:<6} {-best:.3f}")
