# -*- coding: utf-8 -*-
"""Instrument: 区分 POS_LIMIT 与 DELTA_MAX 发散来源 (r0 weak, hessian fixed)。"""
import sys
import numpy as np
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
import nuts
from cr_bayes_model import default_cr_prior, cr_logpost_grad
from sim_grid import rnd_qw, median_scale, cr_ng
from scipy.optimize import minimize

qtrue, ET, BT, BC, n = float(sys.argv[1]) if len(sys.argv) > 1 else 1.45, 100.0, 2.0, 3.0, 200
TA = float(sys.argv[2]) if len(sys.argv) > 2 else 0.85
RHO = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
eta_c = RHO*ET*median_scale(qtrue, BT)/median_scale(1, BC)
rng = np.random.default_rng(2026)
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
H0 = -nuts.numerical_hessian(gradf, xe, 1e-4)

# patch _leapfrog 记录 pos_div 及能量误差
orig = nuts._leapfrog
rec = dict(pos=0, lf=0, maxerr=-1e9, lp_inf=0, g_inf=0, samples=[])
def wrap(x, r, g, lp_in, eps, Sinv, gradf):
    o = orig(x, r, g, lp_in, eps, Sinv, gradf)
    rec['lf'] += 1
    x1, r1, g1, lp1 = o[0], o[1], o[2], o[3]
    L0 = lp_in-0.5*r@(Sinv@r); L1 = lp1-0.5*r1@(Sinv@r1)
    rec['maxerr'] = max(rec['maxerr'], L1-L0)
    if o[-1]:
        rec['pos'] += 1
        if not np.isfinite(lp1): rec['lp_inf'] += 1
        if np.any(~np.isfinite(g1)): rec['g_inf'] += 1
        if len(rec['samples']) < 8:
            qq=[2-np.exp(x1[0]),2-np.exp(x1[3])]
            rec['samples'].append(dict(lp_finite=bool(np.isfinite(lp1)),
                g_bad=[int(i) for i in np.where(~np.isfinite(g1))[0]],
                q=[round(float(v),3) for v in qq],
                eta=[round(float(np.exp(x1[1])),2),round(float(np.exp(x1[4])),2)],
                beta=[round(float(np.exp(x1[2])),3),round(float(np.exp(x1[5])),3)],
                xraw=[round(float(v),2) for v in x1]))
    return o
nuts._leapfrog = wrap

dg = nuts.sample_nuts([xe], gradf, n_draw=500, warmup=200, target_accept=TA,
                     init_hessian=H0)
print(f"[q={qtrue} rho={RHO} target={TA}] leapfrog={rec['lf']} pos_div={rec['pos']} "
      f"lp非有限={rec['lp_inf']} g非有限={rec['g_inf']} 能量误差max={rec['maxerr']:.1f}")
print(f"发散transition: warmup={dg['divergent_warmup']} post={dg['divergent']}/300")
q0 = 2-np.exp(dg["chains"][0, 200:, 0]); q1 = 2-np.exp(dg["chains"][0, 200:, 3])
print(f"目标q0: med={np.median(q0):.3f} sd={q0.std():.3f} "
      f"[{np.percentile(q0,2.5):.2f},{np.percentile(q0,97.5):.2f}]")
print(f"竞争q1: med={np.median(q1):.3f} sd={q1.std():.3f} "
      f"[{np.percentile(q1,2.5):.2f},{np.percentile(q1,97.5):.2f}]")
for s in rec['samples']:
    print(s)
print(f"accept={dg['accept']:.3f} depth={dg['depth']:.2f} elapsed={dg['elapsed']:.1f}s")
