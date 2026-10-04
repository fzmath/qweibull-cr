# -*- coding: utf-8 -*-
"""Cause-specific 可分离拟合: 竞争风险对数似然按模式分解,
每个模式 j 的 MLE = 把'非 j 失效'(竞争失效+右删失)均当右删失的单分布 MLE。
低维、数值稳定; 同时用于模拟验证与真实数据分析。"""
import sys, warnings
import numpy as np
from scipy.optimize import minimize
from scipy import stats
warnings.filterwarnings("ignore")
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
import qweibull as qw

OPT = dict(maxiter=400, ftol=1e-12, gtol=1e-9)


def fit_single_mode(t, status, fix_q, verbose=False):
    """单分布 q-Weibull MLE。status: 1=本模式失效, 0=右删失。"""
    t_med = max(np.median(t), 1e-6)
    q_grid = [1.0] if fix_q else [0.7, 0.85, 1.0, 1.2, 1.45]
    b_grid = [0.7, 1.5, 3.0]
    best = None
    for q0 in q_grid:
        for b0 in b_grid:
            # 数据驱动 lambda 初值 (Weibull 近似), 跨几个量级
            l_center = (1.0 / t_med) ** b0
            for lmult in [0.2, 1.0, 5.0]:
                l0 = l_center * lmult
                if fix_q:
                    theta0 = np.array([np.log(l0), np.log(b0)])
                    r = minimize(lambda th: qw.cr_neg_loglik(th, 1, t, status, True),
                                 theta0, method="L-BFGS-B",
                                 bounds=[(None, None), (None, None)], options=OPT)
                    q_hat = 1.0
                else:
                    theta0 = qw.pack(np.array([q0]), np.array([l0]), np.array([b0]))
                    r = minimize(lambda th: qw.cr_neg_loglik(th, 1, t, status, False),
                                 theta0, method="L-BFGS-B",
                                 bounds=[(qw.xq_bounds()[0], qw.xq_bounds()[1]),
                                         (None, None), (None, None)], options=OPT)
                    q_hat = 2 - np.exp(r.x[0])
                if best is None or r.fun < best.fun:
                    best = r
    if fix_q:
        # q=1 => x_q = log(2-1) = 0
        q_hat, l_hat, b_hat = qw.unpack(np.concatenate([[0.0], best.x]), 1)
    else:
        q_hat, l_hat, b_hat = qw.unpack(best.x, 1)
    q_hat = float(np.atleast_1d(q_hat)[0])
    l_hat = float(np.atleast_1d(l_hat)[0])
    b_hat = float(np.atleast_1d(b_hat)[0])
    return dict(q=q_hat, lam=l_hat, beta=b_hat, logL=float(-best.fun))


def fit_cr(t, cause, K, fix_q):
    """分模式拟合竞争风险, 汇总总 logL。"""
    out, ll = [], 0.0
    for j in range(K):
        status_j = (cause == (j + 1)).astype(int)
        r = fit_single_mode(t, status_j, fix_q)
        out.append(r); ll += r["logL"]
    return out, ll


# ===================== 模拟验证 =====================
if __name__ == "__main__":
    rng = np.random.default_rng(20261002)
    K = 2
    q_true = np.array([1.45, 0.75])
    b_true = np.array([2.00, 3.00])
    l_true = np.array([1.70e-4, 5.18e-7])

    def sim(n=300, censor_at=250.0):
        ts, cs = [], []
        grids = {}
        for j in range(K):
            hi = qw.support_upper(q_true[j], l_true[j], b_true[j])
            if not np.isfinite(hi):
                hi = 3000.0
            gg = np.linspace(1e-6, hi * 0.999, 20000)
            FF = np.clip(qw.cdf(q_true[j], l_true[j], b_true[j], gg), 0, 1)
            grids[j] = (gg, FF)
        for _ in range(n):
            latent = [np.interp(rng.random(), grids[j][1], grids[j][0]) for j in range(K)]
            t = min(latent); c = int(np.argmin(latent)) + 1
            if censor_at is not None and t > censor_at:
                ts.append(censor_at); cs.append(0)
            else:
                ts.append(t); cs.append(c)
        return np.array(ts), np.array(cs)

    t, cause = sim()
    print(f"模拟 n={len(t)}; 模式1={sum(cause==1)}, 模式2={sum(cause==2)}, 删失={sum(cause==0)}")
    rq, llq = fit_cr(t, cause, K, False)
    rw, llw = fit_cr(t, cause, K, True)
    print("\n===== cause-specific q-Weibull MLE =====")
    for j in range(K):
        print(f"  模式{j+1}: q_hat={rq[j]['q']:.3f}(真{q_true[j]})  "
              f"beta_hat={rq[j]['beta']:.3f}(真{b_true[j]})  logL={rq[j]['logL']:.2f}")
    print(f"  总 logL(q)={llq:.3f}")
    print(f"  总 logL(q=1)={llw:.3f}")
    d = 2 * (llq - llw)
    print(f"  LRT 2dLL={d:.3f}, df=2, p={stats.chi2.sf(d,2):.3e}")
    print("  预期: q_hat 接近真值, p<0.05 => cause-specific 分离拟合解决数值问题")
