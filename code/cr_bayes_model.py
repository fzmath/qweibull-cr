# -*- coding: utf-8 -*-
"""q-Weibull 竞争风险 Bayes 后验模型 (K=2 原因)。

变换空间  x = [xq1,xe1,xb1, xq2,xe2,xb2]
    q_j = 2-exp(xq_j), eta_j=exp(xe_j), beta_j=exp(xb_j), lam_j=eta_j^-beta_j
对数似然按原因可分离: 原因 j 子似然
    本原因(cause==j) -> log f_j ; 其余(他因事件 + 右删失) -> log R_j
先验每原因独立, 在 (q, log eta, log beta) 上给定。
数据接口: t, cause (0=E, 1=D, -1=右删失)。
"""
import numpy as np
from bayes_model import Prior, default_prior, logpost_grad

K = 2


def default_cr_prior(t, cause, q_scale=0.5):
    priors = []
    for j in range(K):
        status_j = (cause == j).astype(int)
        priors.append(default_prior(t, status_j, q_scale=q_scale))
    return priors


def cr_logpost_grad(x, t, cause, priors):
    total = 0.0
    grad = np.zeros(3 * K)
    for j in range(K):
        status_j = (cause == j).astype(int)
        xj = x[3 * j:3 * j + 3]
        lp, gj = logpost_grad(xj, t, status_j, priors[j])
        if not np.isfinite(lp):
            return -np.inf, np.zeros(3 * K)
        total += lp
        grad[3 * j:3 * j + 3] = gj
    return total, grad


def cr_logpost(x, t, cause, priors):
    return cr_logpost_grad(x, t, cause, priors)[0]
