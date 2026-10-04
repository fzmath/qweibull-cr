# -*- coding: utf-8 -*-
"""健全性检验: 用已知 q!=1 的 q-Weibull 竞争风险生成数据, 验证 MLE 能识别 q。"""
import sys, warnings
import numpy as np
from scipy.optimize import minimize
from scipy import stats
warnings.filterwarnings("ignore")
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
import qweibull as qw

K = 2
rng = np.random.default_rng(20261002)

# 真实参数: 模式1 强重尾(q>1,峰形), 模式2 强有界(q<1); 两模式中位寿命接近=>比例均衡
q_true = np.array([1.45, 0.75])
b_true = np.array([2.00, 3.00])
l_true = np.array([1.70e-4, 5.18e-7])
print("真实 q:", q_true, " beta:", b_true)


def sim_one(n=120, censor_at=None):
    """生成竞争风险数据: 每个单元各模式潜在时间, 取 min。"""
    ts, cs = [], []
    # 用数值反函数采样: 在网格上做 CDF, 逆变换
    grids = {}
    for j in range(K):
        hi = qw.support_upper(q_true[j], l_true[j], b_true[j])
        if not np.isfinite(hi):
            hi = 2000.0
        g = np.linspace(1e-6, hi * 0.999, 20000)
        F = qw.cdf(q_true[j], l_true[j], b_true[j], g)
        F = np.clip(F, 0, 1)
        grids[j] = (g, F)
    for _ in range(n):
        latent = []
        for j in range(K):
            u = rng.random()
            g, F = grids[j]
            latent.append(np.interp(u, F, g))
        t = min(latent)
        c = int(np.argmin(latent)) + 1
        if censor_at is not None and t > censor_at:
            ts.append(censor_at); cs.append(0)
        else:
            ts.append(t); cs.append(c)
    return np.array(ts), np.array(cs)


t, cause = sim_one(n=300, censor_at=250.0)
print(f"模拟 n={len(t)}; 模式1={np.sum(cause==1)}, 模式2={np.sum(cause==2)}, 删失={np.sum(cause==0)}")

lo_q, hi_q = qw.xq_bounds()
BQ = [(lo_q, hi_q)] * K + [(None, None)] * (2 * K)


def fit(fix_q):
    best = None
    for q0 in ([0.7, 0.9, 1.0, 1.2, 1.4] if not fix_q else [1.0]):
        for b0 in [1.5, 2.5, 3.5]:
            if fix_q:
                theta0 = np.concatenate([np.log(np.ones(K) * 1e-4), np.log(np.ones(K) * b0)])
                r = minimize(qw.cr_neg_loglik, theta0, args=(K, t, cause, True),
                             method="L-BFGS-B", bounds=[(None, None)] * (2 * K))
            else:
                q0v = np.array([q0, q0])
                bv = np.array([b0, b0])
                lv = np.array([1e-3, 1e-8])
                theta0 = qw.pack(q0v, lv, bv)
                r = minimize(qw.cr_neg_loglik, theta0, args=(K, t, cause, False),
                             method="L-BFGS-B", bounds=BQ)
            if best is None or r.fun < best.fun:
                best = r
    return best


rq = fit(False)
rw = fit(True)
q_hat, l_hat, b_hat = qw.unpack(rq.x, K)
print("\n===== q-Weibull CR 拟合 =====")
for j in range(K):
    print(f"  模式{j+1}: q_hat={q_hat[j]:.3f} (真{q_true[j]}), beta_hat={b_hat[j]:.3f} (真{b_true[j]})")
print(f"  logL={-rq.fun:.3f}")
print("===== Weibull CR (q=1) =====")
print(f"  logL={-rw.fun:.3f}")
lrt = 2 * (-rq.fun - (-rw.fun))
print(f"\n  LRT 2dLL={lrt:.3f}, df=2, p={stats.chi2.sf(lrt,2):.3e}")
print("  预期: q_hat 接近真值, LRT 显著(p<0.05) => 代码可识别 q")
