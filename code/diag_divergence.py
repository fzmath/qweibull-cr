# -*- coding: utf-8 -*-
"""诊断 q=1.45 weak 重尾场景发散: 对比 adapt_mass/target_accept 配置。"""
import sys
import numpy as np
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
from cr_bayes_model import default_cr_prior, cr_logpost_grad
from nuts import sample_nuts, numerical_hessian
from sim_grid import rnd_qw, median_scale, cr_ng
from scipy.optimize import minimize

qtrue, ET, BT, BC, n = 1.45, 100.0, 2.0, 3.0, 200
rho = 4.0
eta_c = rho*ET*median_scale(qtrue, BT)/median_scale(1, BC)
WU, PD = 200, 300
rng = np.random.default_rng(2026)
cfgs = [("fixed,.85", False, 0.85), ("adapt,.85", True, 0.85), ("adapt,.95", True, 0.95)]
for r in range(4):
    T0 = rnd_qw(qtrue, ET, BT, n, rng); T1 = rnd_qw(1, eta_c, BC, n, rng)
    t = np.minimum(T0, T1); cause = (T1 < T0).astype(int)
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
    H0 = -numerical_hessian(gradf, xe, 1e-4)
    for nm, am, ta in cfgs:
        dg = sample_nuts([xe], gradf, n_draw=WU+PD, warmup=WU, target_accept=ta,
                         init_hessian=H0, adapt_mass=am)
        print(f"r{r} {nm}: div={dg['divergent']}/{PD} ess_q={dg['ess'][0]:.0f} "
              f"ess/s={dg['ess'].sum()/dg['elapsed']:.1f} depth={dg['depth']:.1f} "
              f"acc={dg['accept']:.2f} t={dg['elapsed']:.1f}s", flush=True)
