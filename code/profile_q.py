# -*- coding: utf-8 -*-
"""单分布 q-Weibull 的 q 剖面诊断 (无竞争/无删失), 定位 q 可识别性。
所有优化加 maxiter 上限, 防止病态似然上卡死。"""
import sys, warnings
import numpy as np
from scipy.optimize import minimize
from scipy import stats
warnings.filterwarnings("ignore")
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
import qweibull as qw

OPT = dict(maxiter=120, ftol=1e-10, gtol=1e-7)
rng = np.random.default_rng(77)
q_t, b_t, l_t = 1.45, 2.0, 1.70e-4
n = 400

# 采样 (逆变换)
hi = qw.support_upper(q_t, l_t, b_t)
if not np.isfinite(hi):
    hi = 3000.0
g = np.linspace(1e-6, hi * 0.999, 30000)
F = np.clip(qw.cdf(q_t, l_t, b_t, g), 0, 1)
t = np.interp(rng.random(n), F, g)
cause = np.ones(n, int)
print(f"单分布采样 n={n}, q真={q_t}, 时间 {t.min():.2f}~{t.max():.2f}, 均值 {t.mean():.2f}")


def nll_qfixed(xlb, q):
    lam, beta = np.exp(xlb[0]), np.exp(xlb[1])
    ll = qw.cr_loglik(np.array([q]), np.array([lam]), np.array([beta]), t, cause)
    return -ll if np.isfinite(ll) else 1e12


# 1) 自由拟合
best = None
for q0 in [0.9, 1.2, 1.45]:
    theta0 = qw.pack(np.array([q0]), np.array([1e-3]), np.array([2.0]))
    r = minimize(lambda th: qw.cr_neg_loglik(th, 1, t, cause, False), theta0,
                 method="L-BFGS-B",
                 bounds=[(qw.xq_bounds()[0], qw.xq_bounds()[1]), (None, None), (None, None)],
                 options=OPT)
    if best is None or r.fun < best.fun:
        best = r
qh = 2 - np.exp(best.x[0])
print(f"自由拟合: q_hat={qh:.3f}, beta_hat={np.exp(best.x[2]):.3f}, logL={-best.fun:.3f}")

# 2) q 剖面 (固定 q, 多初值优化 lambda,beta)
print("\n  q    profile_logL   相对q=1")
prof = {}
for q in np.round(np.arange(0.80, 1.81, 0.05), 2):
    rbest = None
    for l0 in [l_t, 1e-3, 1e-5]:
        r = minimize(nll_qfixed, np.array([np.log(l0), np.log(b_t)]), args=(q,),
                     method="L-BFGS-B", options=OPT)
        if rbest is None or r.fun < rbest.fun:
            rbest = r
    prof[q] = -rbest.fun
ll1 = prof[1.0]
peak_q = max(prof, key=prof.get)
for q, ll in prof.items():
    mark = " <-q=1" if abs(q - 1.0) < 1e-9 else (" <-真值" if abs(q - q_t) < 1e-9 else "")
    print(f" {q:4.2f}   {ll:9.3f}   {ll-ll1:+7.3f}{mark}")
print(f"\n剖面峰值 q*={peak_q} (真值 {q_t}); 峰值-q=1 LL差={prof[peak_q]-ll1:.3f}")
d = 2 * (prof[peak_q] - ll1)
print(f"LRT: 2dLL={d:.3f}, df=1, p={stats.chi2.sf(d,1):.3e}")
