# -*- coding: utf-8 -*-
"""标准 NUTS, H&G (2014) Algorithm 6, slice 采样, 稠密质量矩阵。

通用: gradf(x) -> (logposterior, grad)。
质量矩阵 M 近似后验协方差, 动量 r~N(0,M), M^-1=Sinv。
slice: 固定 logu, 子树以有效点数 n 计数, 候选按 n2/(n1+n2) 概率选取。
外层可经 Whitened 在白化坐标用单位质量采样。
发散检测覆盖能量误差负向 (DELTA_MAX) 与正向 (积分指数不稳定, POS_LIMIT)。
"""
import time, warnings
import numpy as np
warnings.filterwarnings("ignore")

DELTA_MAX = 1000.0
POS_LIMIT = 20.0
J_MAX = 10


# ---------------- 质量矩阵 ----------------
def identity_mass(d):
    return np.eye(d), np.eye(d), np.eye(d)


def mass_from_hessian(H, hess_min=1e-6):
    # H = 负 Hessian = 理想质量矩阵 M (精度)
    d = H.shape[0]
    H = 0.5 * (H + H.T)
    w, Q = np.linalg.eigh(H)
    w = np.clip(w, hess_min, None)
    M = (Q * w) @ Q.T          # 质量矩阵
    Sinv = (Q * (1.0 / w)) @ Q.T   # M^-1
    try:
        return M, Sinv, np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        return identity_mass(d)


def mass_from_samples(X, reg=1e-4):
    # 样本协方差 S; 理想质量 M = S^-1
    d = X.shape[1]
    S = np.cov(X, rowvar=False)
    if np.ndim(S) == 0:
        S = np.atleast_2d(S)
    S = S + (reg * np.trace(S) / d + 1e-8) * np.eye(d)
    try:
        M = np.linalg.inv(S)
        return M, S, np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        return identity_mass(d)


def numerical_hessian(gradf, x, epsilon=1e-4):
    d = x.shape[0]
    H = np.zeros((d, d))
    for k in range(d):
        xp = x.copy(); xp[k] += epsilon
        xm = x.copy(); xm[k] -= epsilon
        _, gp = gradf(xp)
        _, gm = gradf(xm)
        H[:, k] = (gp - gm) / (2 * epsilon)
    return 0.5 * (H + H.T)


def cov_from_hessian(H, hmin=1e-6):
    H = 0.5 * (H + H.T)
    w, Q = np.linalg.eigh(H)
    w = np.clip(w, hmin, None)
    return (Q * (1.0 / w)) @ Q.T


class Whitened:
    """x = xbar + L u (L L^T = C0); 在 u 空间单位质量采样。"""

    def __init__(self, gradf_x, xbar, C0):
        self.gradf_x = gradf_x
        self.xbar = np.asarray(xbar, float)
        self.L = np.linalg.cholesky(C0)

    def gradf(self, u):
        x = self.xbar + self.L @ u
        lp, gx = self.gradf_x(x)
        return lp, self.L.T @ gx

    def to_x(self, u):
        return self.xbar + (u @ self.L.T)

    def to_u(self, x):
        return np.linalg.solve(self.L, np.asarray(x) - self.xbar)


# ---------------- 积分 ----------------
def _leapfrog(x, r, g, lp_in, eps, Sinv, gradf):
    L_in = lp_in - 0.5 * r @ (Sinv @ r)
    r = r + 0.5 * eps * g
    x = x + eps * (Sinv @ r)
    lp, gnew = gradf(x)
    if not np.isfinite(lp) or np.any(~np.isfinite(gnew)):
        return x, r, gnew, lp, True
    r = r + 0.5 * eps * gnew
    if np.any(~np.isfinite(r)):
        return x, r, gnew, lp, True
    L1 = lp - 0.5 * r @ (Sinv @ r)
    return x, r, gnew, lp, bool(L1 - L_in > POS_LIMIT)


def _draw_r(Lc):
    return Lc @ np.random.normal(size=Lc.shape[0])


def _find_reasonable_stepsize(x, Sinv, Lc, gradf):
    lp0, g0 = gradf(x)
    if not np.isfinite(lp0):
        return 1e-3

    def energy(r, lp):
        return lp - 0.5 * r @ (Sinv @ r)

    r0 = _draw_r(Lc)          # H&G: 固定初始动量, 不每次重抽
    E0 = energy(r0, lp0)
    eps = 1.0
    x1, r1, g1, lp1, div1 = _leapfrog(x, r0.copy(), g0, lp0, eps, Sinv, gradf)
    logratio = -np.inf if div1 else energy(r1, lp1) - E0
    a = 1 if logratio > np.log(0.5) else -1

    for _ in range(50):
        eps *= 2.0 if a == 1 else 0.5
        x1, r1, g1, lp1, div1 = _leapfrog(x, r0.copy(), g0, lp0, eps, Sinv, gradf)
        logratio = -np.inf if div1 else energy(r1, lp1) - E0
        # 目标: 单步接受概率 ~0.5 (ratio=log0.5)
        if a == 1:                          # doubling: 概率跌破0.5停
            if logratio < np.log(0.5):
                break
        else:                               # halving: 概率升到0.5停
            if logratio >= np.log(0.5):
                break
        if eps < 1e-7 or eps > 1e3:
            break
    return max(eps, 1e-7)


# ---------------- 递归树 (slice) ----------------
class Sub:
    __slots__ = ("xm", "rm", "gm", "lm", "xp", "rp", "gp", "lp",
                 "xcand", "n", "stop", "alpha", "nalpha", "ndiv", "leaves")


def _build(x, r, g, lp_in, logu, v, j, eps, Sinv, gradf, L0):
    if j == 0:
        x1, r1, g1, lp1, pos_div = _leapfrog(
            x, r, g, lp_in, v * eps, Sinv, gradf)
        s = Sub()
        s.xm = s.xp = x1
        s.rm = s.rp = r1
        s.gm = s.gp = g1
        s.lm = s.lp = lp1
        if pos_div or not np.isfinite(lp1):
            s.xcand, s.n = x1, 0
            s.stop = False
            s.alpha, s.nalpha, s.ndiv = 0.0, 1, 1
            s.leaves = [x1]
            return s
        L1 = lp1 - 0.5 * r1 @ (Sinv @ r1)
        valid = logu <= L1
        s.xcand, s.n = x1, (1 if valid else 0)
        inside = logu < L1 + DELTA_MAX
        s.stop = bool(inside)
        s.alpha = min(1.0, np.exp(min(0.0, L1 - L0)))
        s.nalpha = 1
        s.ndiv = 0 if inside else 1
        s.leaves = [x1]
        return s

    s1 = _build(x, r, g, lp_in, logu, v, j - 1, eps, Sinv, gradf, L0)
    if v == -1:
        s2 = _build(s1.xm, s1.rm, s1.gm, s1.lm, logu, v, j - 1, eps, Sinv, gradf, L0)
        xm, rm, gm, lm = s2.xm, s2.rm, s2.gm, s2.lm
        xp, rp, gp, lpp = s1.xp, s1.rp, s1.gp, s1.lp
    else:
        s2 = _build(s1.xp, s1.rp, s1.gp, s1.lp, logu, v, j - 1, eps, Sinv, gradf, L0)
        xm, rm, gm, lm = s1.xm, s1.rm, s1.gm, s1.lm
        xp, rp, gp, lpp = s2.xp, s2.rp, s2.gp, s2.lp

    s = Sub()
    s.xm, s.rm, s.gm, s.lm = xm, rm, gm, lm
    s.xp, s.rp, s.gp, s.lp = xp, rp, gp, lpp
    s.n = s1.n + s2.n
    if s2.n > 0 and (s1.n == 0 or np.random.random() < s2.n / s.n):
        s.xcand = s2.xcand
    else:
        s.xcand = s1.xcand
    s.alpha = s1.alpha + s2.alpha
    s.nalpha = s1.nalpha + s2.nalpha
    s.ndiv = s1.ndiv + s2.ndiv
    s.leaves = s1.leaves + s2.leaves
    dx = s.xp - s.xm
    turn = (dx @ (Sinv @ s.rm) >= 0) and (dx @ (Sinv @ s.rp) >= 0)
    s.stop = bool(s1.stop and s2.stop and turn)
    return s


def _one_transition(x, g, lp, eps, Sinv, Lc, gradf):
    r0 = _draw_r(Lc)
    L0 = lp - 0.5 * r0 @ (Sinv @ r0)
    logu = L0 + np.log(np.random.random())
    xm = xp = x
    rm = rp = r0
    gm, gp = g, g
    lpm = lpp = lp
    j, keep = 0, True
    xcand, n_tot = x, 1          # 起点计入 (H&G: n=1, C 初始含起点)
    alpha_i, nalpha_i, ndiv_i = 0.0, 0, 0
    while keep and j < J_MAX:
        v = 1 if np.random.random() < 0.5 else -1
        tr = _build(xm if v == -1 else xp,
                    rm if v == -1 else rp,
                    gm if v == -1 else gp,
                    lpm if v == -1 else lpp,
                    logu, v, j, eps, Sinv, gradf, L0)
        n_old = n_tot
        if tr.stop and tr.n > 0 and np.random.random() < min(1.0, tr.n / n_old):
            xcand = tr.xcand
        n_tot += tr.n
        if v == -1:
            xm, rm, gm, lpm = tr.xm, tr.rm, tr.gm, tr.lm
        else:
            xp, rp, gp, lpp = tr.xp, tr.rp, tr.gp, tr.lp
        alpha_i += tr.alpha
        nalpha_i += tr.nalpha
        ndiv_i += tr.ndiv
        dx = xp - xm
        keep = bool(tr.stop
                    and dx @ (Sinv @ rm) >= 0
                    and dx @ (Sinv @ rp) >= 0)
        j += 1
    moved = np.any(xcand != x)
    if moved:
        x = xcand
        lp, g = gradf(x)
    return x, g, lp, j - 1, alpha_i, max(nalpha_i, 1), ndiv_i


# ---------------- 单链 ----------------
def _chain(x0, n_draw, warmup, gradf, target_accept, seed,
           C0, Sinv0, Lc0, adapt_mass=False, verbose=False):
    np.random.seed(seed)
    d = x0.shape[0]
    C, Sinv, Lc = C0, Sinv0, Lc0
    eps = _find_reasonable_stepsize(x0, Sinv, Lc, gradf)

    def _cap(e, Mmat):                     # 理论稳定上限 cap
        try:
            lmax = float(np.linalg.eigvalsh(Mmat).max())
            return min(e, 0.9 * 2.0 / np.sqrt(max(lmax, 1e-12)))
        except Exception:
            return e
    eps = _cap(eps, C)

    gamma, t0da, kappa = 0.05, 10.0, 0.75
    mu, H_bar, log_eps_bar = np.log(10 * eps), 0.0, np.log(eps)
    m_loc = 0

    # dense mass 适应 (Stan 风格, 简化): 更新点 -> 样本窗口 [a,b)
    mass_updates = {75: (25, 75), 150: (75, 150)} if adapt_mass else {}
    buf = []

    x = x0.copy()
    lp, g = gradf(x)
    chain = np.zeros((n_draw, d))
    depths = np.zeros(n_draw)
    alpha_sum, nalpha_tot = 0.0, 0
    ndiv_w, ndiv_p = 0, 0

    for i in range(n_draw):
        x, g, lp, dep, ai, nai, ndi = _one_transition(x, g, lp, eps, Sinv, Lc, gradf)
        chain[i] = x
        depths[i] = dep
        alpha_sum += ai
        nalpha_tot += nai
        dtrans = 1 if ndi > 0 else 0     # 发散 transition (每次迭代至多1, Stan口径)
        if i < warmup:
            ndiv_w += dtrans
        else:
            ndiv_p += dtrans

        if i < warmup:
            a_avg = ai / nai
            m_loc += 1
            H_bar = (1 - 1 / (m_loc + t0da)) * H_bar + (1 / (m_loc + t0da)) * (target_accept - a_avg)
            log_eps = mu - np.sqrt(m_loc) / gamma * H_bar
            eps = max(np.exp(log_eps), 1e-7)
            log_eps_bar = m_loc ** (-kappa) * log_eps + (1 - m_loc ** (-kappa)) * log_eps_bar

            if adapt_mass:
                buf.append(x)
                if (i + 1) in mass_updates:
                    a, b = mass_updates[i + 1]
                    Xw = np.array(buf[a:b])
                    if len(Xw) > d + 2:
                        C, Sinv, Lc = mass_from_samples(Xw)
                        eps = _cap(_find_reasonable_stepsize(x, Sinv, Lc, gradf), C)
                        mu, H_bar, log_eps_bar = np.log(10 * eps), 0.0, np.log(eps)
                        m_loc = 0
                        if verbose:
                            print(f"  [mass@{i+1}] eps={eps:.4f}")
        elif i == warmup:
            eps = max(np.exp(log_eps_bar), 1e-7)
            if verbose:
                print(f"  [post起点] eps={eps:.4f}")

    return chain, depths, alpha_sum / max(nalpha_tot, 1), ndiv_w, ndiv_p


# ---------------- 多链 + 诊断 ----------------
def sample_nuts(theta_inits, gradf, n_draw=1000, warmup=500, target_accept=0.85,
                init_hessian=None, adapt_mass=False):
    d = theta_inits[0].shape[0]
    ngrad = [0]
    _gf = gradf

    def gradf(x):                 # 梯度评估计数包装
        ngrad[0] += 1
        return _gf(x)

    if init_hessian is not None:
        C0, Sinv0, Lc0 = mass_from_hessian(init_hessian)
    else:
        C0, Sinv0, Lc0 = identity_mass(d)

    t0 = time.time()
    chains, depths, accs = [], [], []
    div_w, div_p = [], []
    base = np.asarray(theta_inits[0], float)
    for ci, th0 in enumerate(theta_inits):
        th0 = np.asarray(th0, float)
        lp0, _ = gradf(th0)
        sc = 0.3
        while not np.isfinite(lp0) and sc > 1e-3:          # 非法初始点: 递减扰动重试
            np.random.seed(7000 + ci)
            th0 = base + sc * np.random.standard_normal(d)
            lp0, _ = gradf(th0)
            sc *= 0.5
        if not np.isfinite(lp0):                           # 最终回退到 MLE/众数
            th0 = base
        ch, dep, acc, ndw, ndp = _chain(
            th0, n_draw, warmup, gradf, target_accept, seed=900 + ci,
            C0=C0, Sinv0=Sinv0, Lc0=Lc0, adapt_mass=adapt_mass)
        chains.append(ch); depths.append(dep); accs.append(acc)
        div_w.append(ndw); div_p.append(ndp)
    chains = np.array(chains)
    post = chains[:, warmup:, :]
    diag = dict(
        rhat=_rhat(post), ess=_ess(post),
        mean=post.mean(axis=(0, 1)),
        ci=np.quantile(post, [0.025, 0.975], axis=(0, 1)),
        accept=float(np.mean(accs)),
        depth=float(np.mean([d[warmup:].mean() for d in depths])),
        divergent=int(np.sum(div_p)),
        divergent_warmup=int(np.sum(div_w)),
        chains=chains,
    )
    diag["n_grad"] = int(ngrad[0])
    diag["elapsed"] = time.time() - t0
    diag["ess_per_sec"] = diag["ess"].sum() / diag["elapsed"]
    return diag


def _rhat(post):
    C, N, d = post.shape
    means = post.mean(axis=1)
    vars_ = post.var(axis=1, ddof=1)
    B = N * np.var(means, axis=0, ddof=1)
    W = vars_.mean(axis=0)
    varhat = (N - 1) / N * W + B / N
    return np.sqrt(varhat / np.where(W > 0, W, np.nan))


def _ess(post):
    C, N, d = post.shape
    ess = np.zeros(d)
    for k in range(d):
        acfs = []
        for c in range(C):
            x = post[c, :, k] - post[c, :, k].mean()
            f = np.fft.rfft(x, n=2 * N)
            ac = np.fft.irfft(f * np.conj(f))[:N]
            ac /= ac[0]
            acfs.append(ac)
        acf = np.mean(acfs, axis=0)
        tau = 1.0
        for lag in range(1, min(N - 3, 1000), 2):
            pair = acf[lag] + acf[lag + 1]
            if pair < 0:
                break
            tau += 2 * pair
        ess[k] = C * N / tau
    return ess
