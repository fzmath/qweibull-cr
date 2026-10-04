# -*- coding: utf-8 -*-
"""
q-Weibull 分布与竞争风险模型 (Python, 与 MATLAB 代码数学定义一致)。

f(t) = (2-q)*beta*lam*t**(beta-1) * [1-(1-q)*lam*t**beta]**(1/(1-q))
R(t) = [1-(1-q)*lam*t**beta]**((2-q)/(1-q))
lam = eta**(-beta); q<1 有界, q=1 标准 Weibull, 1<q<2 重尾。

竞争风险 (k 模式串联), T=min(T_1,...,T_k):
    因模式 j 失效: f_j(t) * prod_{l!=j} R_l(t)
    右删失:        prod_j R_j(t)
"""
import numpy as np

# q 的合理范围 (用于无约束变换的边界); q<2 数学合法, 上界取 1.95 保数值安全
Q_MIN, Q_MAX = 0.20, 1.95


def _inner(q, lam, beta, t):
    tb = np.power(t, beta)
    return 1.0 - (1.0 - q) * lam * tb


def log_pdf(q, lam, beta, t):
    t = np.asarray(t, dtype=float)
    scalar = t.ndim == 0
    t = np.atleast_1d(t)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if abs(q - 1.0) < 1e-12:
            out = np.log(beta) + np.log(lam) + (beta - 1.0) * np.log(t) - lam * t ** beta
            out = np.where(t > 0.0, out, -np.inf)
        else:
            inner = _inner(q, lam, beta, t)
            out = (np.log(2.0 - q) + np.log(beta) + np.log(lam)
                   + (beta - 1.0) * np.log(t)
                   + (1.0 / (1.0 - q)) * np.log(inner))
            out = np.where((inner > 0.0) & (t > 0.0), out, -np.inf)
    return out[0] if scalar else out


def log_sf(q, lam, beta, t):
    t = np.asarray(t, dtype=float)
    scalar = t.ndim == 0
    t = np.atleast_1d(t)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if abs(q - 1.0) < 1e-12:
            out = -lam * t ** beta
        else:
            inner = _inner(q, lam, beta, t)
            out = ((2.0 - q) / (1.0 - q)) * np.log(inner)
            out = np.where((inner > 0.0) & (t >= 0.0), out, -np.inf)
            out = np.where(t == 0.0, 0.0, out)
    return out[0] if scalar else out


def pdf(q, lam, beta, t):
    return np.exp(log_pdf(q, lam, beta, t))


def sf(q, lam, beta, t):
    return np.exp(log_sf(q, lam, beta, t))


def cdf(q, lam, beta, t):
    return 1.0 - sf(q, lam, beta, t)


def support_upper(q, lam, beta):
    return np.inf if q >= 1.0 else (1.0 / ((1.0 - q) * lam)) ** (1.0 / beta)


# ---- 无约束变换: q=2-exp(xq), lam=exp(xl), beta=exp(xb) ----
def unpack(theta, k):
    q = 2.0 - np.exp(theta[0:k])
    lam = np.exp(theta[k:2 * k])
    beta = np.exp(theta[2 * k:3 * k])
    return q, lam, beta


def pack(q, lam, beta):
    return np.concatenate([np.log(2.0 - q), np.log(lam), np.log(beta)])


def xq_bounds():
    """x_q 的边界, 对应 q in [Q_MIN, Q_MAX]。"""
    return (np.log(2.0 - Q_MAX), np.log(2.0 - Q_MIN))


def _safe_sum(parts, axis):
    """沿轴求和; 任一元素为 -inf (非有限) 则结果 -inf, 避免 inf-inf。"""
    finite = np.isfinite(parts).all(axis=axis)
    s = np.where(finite, np.where(np.isfinite(parts), parts, 0.0).sum(axis=axis), -np.inf)
    return s


def cr_loglik(q, lam, beta, t, cause):
    """向量化竞争风险对数似然 (原始参数)。"""
    t = np.asarray(t, float)
    cause = np.asarray(cause, int)
    k = len(q)
    n = len(t)
    logR = np.empty((k, n))
    logf = np.empty((k, n))
    for j in range(k):
        logR[j] = log_sf(q[j], lam[j], beta[j], t)
        logf[j] = log_pdf(q[j], lam[j], beta[j], t)

    contrib = np.empty(n)
    for i in range(n):
        ci = cause[i]
        if ci == 0:
            contrib[i] = _safe_sum(logR[:, i], axis=0)
        else:
            c = ci - 1
            others = np.delete(logR[:, i], c)
            r_other = _safe_sum(others, axis=0)
            contrib[i] = logf[c, i] + r_other
    return contrib.sum()


def cr_neg_loglik(theta, k, t, cause, fix_q_one=False):
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if fix_q_one:
            q = np.ones(k)
            lam = np.exp(theta[0:k])
            beta = np.exp(theta[k:2 * k])
        else:
            q, lam, beta = unpack(theta, k)
        ll = cr_loglik(q, lam, beta, t, cause)
    if not np.isfinite(ll):
        return 1e10
    return -ll
