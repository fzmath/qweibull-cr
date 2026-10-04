# -*- coding: utf-8 -*-
"""Fig6: 强/中/弱截断 q 剖面似然 (竞争原因固定标准Weibull, 只profile目标q)。"""
import sys
import numpy as np
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qweibull import Q_MIN, Q_MAX
from bayes_model import _loglik_and_grad_raw
from scipy.optimize import minimize

def rnd_qw(q, eta, beta, n, rng):
    u = np.maximum(rng.random(n), 1e-12); lam = eta**-beta
    if abs(q-1) < 1e-9:
        return eta*(-np.log(u))**(1/beta)
    d = 1-q
    return ((1-u**(1/((2-q)/d)))/(d*lam))**(1/beta)

def make_obj(qfix, t, cause):
    # 自由 f=[目标xl,xb, 竞争xl,xb]; theta 竞争xq固定0(q=1)
    def obj(f):
        theta = np.array([np.log(2-qfix), f[0], f[1], 0.0, f[2], f[3]])
        neg = 0.0; g = np.zeros(6)
        for j in range(2):
            xq, xl, xb = theta[3*j:3*j+3]
            q = 2-np.exp(xq); lam = np.exp(xl); beta = np.exp(xb)
            if q < Q_MIN or q > Q_MAX:
                return 1e10, np.full(4, 1e6)
            ll, gq, gl, gb = _loglik_and_grad_raw(
                q, lam, beta, t, (cause == j).astype(int))
            if not np.isfinite(ll):
                return 1e10, np.full(4, 1e6)
            neg += ll
            g[3*j+1] = gl*lam; g[3*j+2] = gb*beta
        gf = np.array([g[1], g[2], g[4], g[5]])
        return -neg, -gf
    return obj

BND = [(-25, 5), (-2, 3), (-25, 5), (-2, 3)]
def profile_point(qfix, f0, t, cause):
    obj = make_obj(qfix, t, cause)
    r = minimize(obj, f0, jac=True, method="L-BFGS-B", bounds=BND,
                 options=dict(maxiter=500, ftol=1e-10, gtol=1e-7))
    if not r.success or np.max(np.abs(r.jac)) > 1e-3:
        r2 = minimize(obj, f0 + np.random.default_rng(0).normal(0, .3, 4),
                      jac=True, method="L-BFGS-B", bounds=BND,
                      options=dict(maxiter=800))
        if r2.fun < r.fun:
            r = r2
    return r.fun

plt.rcParams.update({"font.size": 13, "axes.grid": True, "grid.alpha": .3,
                     "grid.linestyle": "--", "figure.dpi": 150})
qs = np.arange(1.00, 1.71, 0.05)
conds = [("Strong truncation", 68.0, "#c0392b"),
         ("Moderate truncation", 113.0, "#e08e0b"),
         ("Weak truncation", 452.0, "#27795b")]
fig, ax = plt.subplots(figsize=(8.4, 5.2))
for name, eta2, col in conds:
    rng = np.random.default_rng(2026)
    T0 = rnd_qw(1.45, 76.7, 2, 200, rng); T1 = rnd_qw(1, eta2, 3, 200, rng)
    t = np.minimum(T0, T1); cause = (T1 < T0).astype(int)
    f0 = np.array([np.log(76.7**-2), np.log(2),
                   np.log(eta2**-3), np.log(3)])
    vals = np.array([profile_point(qf, f0, t, cause) for qf in qs])
    vals -= vals.min()
    ax.plot(qs, -vals, "-o", color=col, ms=4.5, lw=1.9, label=name)
ax.axvline(1.45, color="black", ls=":", lw=1.5)
ax.text(1.46, ax.get_ylim()[1]*.9, "true $q=1.45$", fontsize=11)
ax.set_xlabel("$q$"); ax.set_ylabel("Profile log-likelihood (relative to peak)")
ax.set_title("Identifiability of $q$ under competing truncation")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(r"G:\OPT\q-Weibull竞争风险论文\figures\fig6_profile_q.png",
            bbox_inches="tight")
fig.savefig(r"G:\OPT\q-Weibull竞争风险论文\figures\fig6_profile_q.pdf",
            bbox_inches="tight")
print("Fig6 saved")
