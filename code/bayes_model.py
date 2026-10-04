# -*- coding: utf-8 -*-
"""q-Weibull (单分布/竞争风险分模式) Bayes 后验模型。

无约束/变换空间  x = [x_q, x_e, x_b]:
    q = 2 - exp(x_q),  eta = exp(x_e),  beta = exp(x_b)
    lambda = eta^(-beta)          (派生量)
似然在变换参数下表达 (不含 Jacobian); 先验在变换空间指定 (含 Jacobian)。

先验在自然且近似独立的 (q, log eta, log beta) 上给定 (数值稳定):
    q       ~ Normal(mu_q, sigma_q) 截断[Q_MIN,Q_MAX]  (q 空间, 需 Jacobian x_q)
    log eta ~ Normal(mu_e, sigma_e)
    log beta~ Normal(mu_b, sigma_b)
注: 不在 (log lambda, log beta) 给先验, 因 log lambda=-beta log eta 与 beta
强耦合, 会在 beta 偏离粗值时经 log lambda 项任意主导推断。
"""
import numpy as np
from qweibull import log_pdf, log_sf, Q_MIN, Q_MAX

_EPS = 1e-9
_SOFT = 1e-3   # soft barrier 阈值; 边界曲率由 1/_SOFT 限制 (有限), 消除硬墙死锁


def _slog(u):
    """光滑 log: u>_SOFT 用 log(u); u<=_SOFT 线性延伸 (导数=1/_SOFT 有限)。"""
    return np.where(u > _SOFT, np.log(np.maximum(u, 1e-300)),
                    np.log(_SOFT) + (u - _SOFT) / _SOFT)


def _slogp(u):
    """d/du _slog。"""
    return np.where(u > _SOFT, 1.0 / np.maximum(u, 1e-300), 1.0 / _SOFT)


class Prior:
    def __init__(self, mu_q=1.0, sigma_q=0.3, mu_e=5.0, sigma_e=1.5,
                 mu_b=0.4, sigma_b=1.2, q_lo=None, q_hi=None, uniform=False):
        self.mu_q, self.sigma_q = mu_q, sigma_q
        self.mu_e, self.sigma_e = mu_e, sigma_e
        self.mu_b, self.sigma_b = mu_b, sigma_b
        self.q_lo = Q_MIN if q_lo is None else q_lo
        self.q_hi = 1.98 if q_hi is None else q_hi
        self.uniform = uniform


def default_prior(t, status, q_center=1.0, q_scale=0.3):
    """数据驱动弱信息先验: log eta 中心=事件中位, log beta 中心=log1.5。"""
    tev = t[status == 1]
    t_use = tev if len(tev) > 2 else t
    med = max(np.median(t_use), 1e-6)
    b0 = 1.5
    return Prior(mu_q=q_center, sigma_q=q_scale,
                 mu_e=np.log(med), sigma_e=1.5,
                 mu_b=np.log(b0), sigma_b=1.2)


def _soft_z(lz, CUT=20.0):
    """z=exp(lz) 的光滑化: lz<CUT 用 exp; 其后线性外推 (有限梯度, C1连续)。
    返回 (z_eff, dz_eff/dlz), 避免 -z 在大 z 时溢出为 inf。"""
    lz = np.asarray(lz, float)
    ec = np.exp(CUT)
    z_eff = np.where(lz < CUT, np.exp(np.minimum(lz, CUT)), ec*(1.0+(lz-CUT)))
    dz = np.where(lz < CUT, np.exp(np.minimum(lz, CUT)), ec)
    return z_eff, dz


def _loglik_and_grad_raw(q, lam, beta, t, status):
    """返回 loglik 及对原始参数 (q, lam, beta) 的梯度; 全样本统一计算。
    z 经 _soft_z 光滑化, 保证参数极端区后验有限且梯度连续。"""
    t = np.asarray(t, float)
    ev = status == 1
    ce = ~ev
    ne = int(ev.sum())
    logt = np.log(np.maximum(t, 1e-300))
    loglam = np.log(lam)
    lz = loglam + beta*logt              # log z, 始终有限
    z_eff, dz = _soft_z(lz)

    if abs(1.0 - q) < _EPS:
        # ---- 标准 Weibull ----
        ll = ne * (np.log(beta) + loglam) + np.sum((beta - 1) * logt[ev]) - np.sum(z_eff)
        if not np.isfinite(ll):
            return -np.inf, np.nan, np.nan, np.nan
        g_q = np.sum(z_eff[ev] ** 2 / 2.0 - 1.0) + np.sum(z_eff[ce] + z_eff[ce] ** 2 / 2.0)
        g_l = ne / lam - np.sum(dz) / lam
        g_b = ne / beta + np.sum(logt[ev]) - np.sum(dz * logt)
        return ll, g_q, g_l, g_b

    d = 1.0 - q
    p = (2.0 - q) / d
    inner = 1.0 - d * z_eff
    li = _slog(inner)
    lip = _slogp(inner)

    ll = (ne * (np.log(2.0 - q) + np.log(beta) + loglam)
          + np.sum((beta - 1) * logt[ev])
          + np.sum(li[ev] / d) + np.sum(p * li[ce]))

    g_q = ((-ne / (2.0 - q))
           + np.sum(li[ev] / d ** 2) + np.sum(z_eff[ev] / d * lip[ev])
           + np.sum(li[ce] / d ** 2) + np.sum(p * z_eff[ce] * lip[ce]))
    g_l = (ne / lam
           - np.sum(dz[ev] / lam * lip[ev])
           - np.sum(p * d * dz[ce] / lam * lip[ce]))
    g_b = (ne / beta + np.sum(logt[ev])
           - np.sum(dz[ev] * logt[ev] * lip[ev])
           - np.sum(p * d * dz[ce] * logt[ce] * lip[ce]))
    return ll, g_q, g_l, g_b


def _xq_softwall(xq, q_lo, q_hi, w=0.3):
    """直接在采样坐标 xq 上加软墙 (q=2-exp(xq))。
    q 有用范围 [q_lo,q_hi] 对应 xq in [XQ_LO,XQ_HI]; 越界给二次排斥。
    在 xq 空间加墙可避免 q->2 (xq->-inf) 时 q 空间墙梯度消失。返回 (V,dV/dxq)。"""
    XQ_LO = np.log(2.0-q_hi)
    XQ_HI = np.log(2.0-q_lo)
    V, dV = 0.0, 0.0
    if xq < XQ_LO:
        e = xq-XQ_LO; V -= 0.5*(e/w)**2; dV -= e/w**2
    if xq > XQ_HI:
        e = xq-XQ_HI; V -= 0.5*(e/w)**2; dV -= e/w**2
    return V, dV


def logpost_grad(x, t, status, prior):
    """变换空间对数后验及其梯度。x=[xq,xe,xb]。"""
    xq, xe, xb = x
    q = 2.0 - np.exp(xq)
    eta = np.exp(xe)
    beta = np.exp(xb)
    # q 越界由 _q_softwall 光滑排斥; beta,eta=exp 恒正, 无硬墙
    lam = eta ** (-beta)
    ll, gq, gl, gb = _loglik_and_grad_raw(q, lam, beta, t, status)
    if not np.isfinite(ll):
        return -np.inf, np.zeros(3)

    # ---- xq 软墙 (直接采样坐标; Jacobian 已含在先验 +xq) ----
    Vq, dVxq = _xq_softwall(xq, prior.q_lo, prior.q_hi)

    # ---- q 先验: 高斯, 或均匀 (仅 Jacobian +xq) ----
    if prior.uniform:
        lp_q = xq
    else:
        lp_q = -0.5 * ((q - prior.mu_q) / prior.sigma_q) ** 2 + xq
    lp_e = -0.5 * ((xe - prior.mu_e) / prior.sigma_e) ** 2
    lp_b = -0.5 * ((xb - prior.mu_b) / prior.sigma_b) ** 2
    logpost = ll + Vq + lp_q + lp_e + lp_b

    # ---- 链式: lambda=eta^-beta
    # dlam/deta = -beta lam/eta ; dlam/dbeta = -lam log eta = -lam xe
    g_xe = -beta * lam * gl                      # 似然对 xe
    g_xb = beta * (gb - gl * lam * xe)           # 似然对 xb
    dq_dxq = -np.exp(xq)
    if prior.uniform:
        glp_xq = 1.0
    else:
        glp_xq = (q - prior.mu_q) / prior.sigma_q ** 2 * np.exp(xq) + 1.0
    glp_xe = -(xe - prior.mu_e) / prior.sigma_e ** 2
    glp_xb = -(xb - prior.mu_b) / prior.sigma_b ** 2

    grad = np.array([
        gq * dq_dxq + dVxq + glp_xq,
        g_xe + glp_xe,
        g_xb + glp_xb,
    ])
    return logpost, grad


def logpost(x, t, status, prior):
    return logpost_grad(x, t, status, prior)[0]
