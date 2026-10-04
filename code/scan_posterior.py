# -*- coding: utf-8 -*-
"""扫描 weak q1.45 竞争模式 eta_D/beta_D 方向 logpost, 查数值平台/突变/下溢。"""
import sys
import numpy as np
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
from cr_bayes_model import default_cr_prior, cr_logpost_grad
from sim_grid import rnd_qw, median_scale, cr_ng
from nuts import numerical_hessian
from scipy.optimize import minimize

qtrue, ET, BT, BC, n = 1.45, 100.0, 2.0, 3.0, 200
eta_c = 4.0*ET*median_scale(qtrue, BT)/median_scale(1, BC)
rng = np.random.default_rng(2026)
T0 = rnd_qw(qtrue, ET, BT, n, rng); T1 = rnd_qw(1, eta_c, BC, n, rng)
t = np.minimum(T0, T1); cause = (T1 < T0).astype(int)
print("D(竞争)事件数 =", int((cause == 1).sum()), " 目标事件 =", int((cause == 0).sum()))
x0 = np.concatenate([[np.log(2-qtrue), np.log(ET**-BT), np.log(BT)],
                     [np.log(1), np.log(eta_c**-BC), np.log(BC)]])
th = minimize(lambda z: cr_ng(z, t, cause), x0, jac=True, method="BFGS",
              options=dict(maxiter=500)).x
priors = default_cr_prior(t, cause)
gradf = lambda x: cr_logpost_grad(x, t, cause, priors)
xe = []
for j in range(2):
    xq, xl, xb = th[3*j:3*j+3]; ll = np.exp(xl); bb = np.exp(xb)
    xe += [xq, np.log(ll**(-1/bb)), xb]
xe = np.array(xe)
# 坐标: [xq0,xe0,xb0, xq1,xe1,xb1] -> 竞争 eta=idx4, beta=idx5
for idx, nm, rng_ in [(4, "eta_D(log)", 8), (5, "beta_D(log)", 6)]:
    base = xe[idx]
    print(f"\n--- 沿 {nm} (MLE值={base:.3f}) ---")
    for d in np.linspace(-rng_, rng_, 13):
        x = xe.copy(); x[idx] = base+d
        lp, g = gradf(x)
        print(f" d={d:+.1f} val={x[idx]:7.2f} logpost={lp:12.3f} g[{idx}]={g[idx]:10.3f}")
