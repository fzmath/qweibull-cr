# -*- coding: utf-8 -*-
"""P1 诊断: 定位 voltage q-W 竞争风险 NUTS 的 post-warmup divergences。

照 run_voltage_bayes.py 完全相同的设置 (6维, 2链, n_draw=2000, warmup=1000,
target_accept=0.9, Hessian 初始质量矩阵, 同种子) 重跑, 并:
  1) 区分 warmup / post-warmup divergences (nuts 内部已分别计数);
  2) monkeypatch nuts._leapfrog 捕获每个发散叶节点的位置 x;
  3) 薄复刻 _chain 主循环, 把发散归因到 (chain, iter), 并记录每步能量 E;
  4) 对每个 post-warmup 发散点分类: near-wall/barrier (xq 越过软墙区间
     [log(2-q_hi), log(2-q_lo)] 或 inner u=1-(1-q)lam t^beta 逼近 _SOFT)
     vs 后验高密度区 (马氏距离小);
  5) 报 max tree depth / mean accept / per-param ESS / ESS-per-s / BFMI;
  6) 顺便验证数据支持区的最小 inner u。
结果写 data/divergence_diag.json。不改 nuts.py / qweibull.py 等核心文件。
"""
import sys, json, time, os
import numpy as np
import pandas as pd
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")

import nuts
from nuts import (_draw_r, _find_reasonable_stepsize, mass_from_hessian,
                  mass_from_samples, numerical_hessian, _ess, _rhat,
                  DELTA_MAX, POS_LIMIT, J_MAX)
from cr_bayes_model import default_cr_prior, cr_logpost_grad
from bayes_model import _SOFT

CODE = r"G:\OPT\q-Weibull竞争风险论文\code"
DATA = r"G:\OPT\q-Weibull竞争风险论文\data"

# ---------------- 1. 数据与模型 (照 run_voltage_bayes.py) ----------------
df = pd.read_csv(DATA + r"\voltage.csv")
t = df["hours"].to_numpy(float)
fm = df["failure_mode"].to_numpy(str)
cause = np.full(len(df), -1)
cause[fm == "E"] = 0
cause[fm == "D"] = 1
N_E = int((cause == 0).sum())
N_D = int((cause == 1).sum())
N_C = int((cause == -1).sum())

priors = default_cr_prior(t, cause)
Q_LO = priors[0].q_lo          # 0.2
Q_HI = priors[0].q_hi          # 1.98
XQ_LO = np.log(2.0 - Q_HI)     # xq 软墙下界
XQ_HI = np.log(2.0 - Q_LO)     # xq 软墙上界

ngrad = [0]
def gradf(x):
    ngrad[0] += 1
    return cr_logpost_grad(x, t, cause, priors)

mle = {
    0: dict(q=1.2414715, lam=0.01414739, beta=0.6550399),
    1: dict(q=0.9345138, lam=1.5741296e-14, beta=5.4219912),
}
def to_x(m):
    eta = m["lam"] ** (-1.0 / m["beta"])
    return np.array([np.log(2 - m["q"]), np.log(eta), np.log(m["beta"])])
x_mle = np.concatenate([to_x(mle[0]), to_x(mle[1])])
rng = np.random.RandomState(7)
inits = [x_mle, x_mle + 0.3 * rng.standard_normal(6)]
H0 = -numerical_hessian(gradf, x_mle, epsilon=1e-4)
C0, Sinv0, Lc0 = mass_from_hessian(H0)

# ---------------- 2. 发散叶节点记录器 (monkeypatch _leapfrog) ----------------
LEAVES = []   # 当前 transition 内所有被判为发散的 leapfrog 叶节点
_orig_leapfrog = nuts._leapfrog

def _rec_leapfrog(x, r, g, lp_in, eps, Sinv, gf):
    out = _orig_leapfrog(x, r, g, lp_in, eps, Sinv, gf)
    x1, r1, g1, lp1, pos_div = out
    L0 = lp_in - 0.5 * r @ (Sinv @ r)
    L1 = lp1 - 0.5 * r1 @ (Sinv @ r1) if np.isfinite(lp1) else -np.inf
    reasons = []
    if pos_div:
        reasons.append("POS_LIMIT")
    if not np.isfinite(lp1):
        reasons.append("lp_nonfinite")
    if np.any(~np.isfinite(g1)):
        reasons.append("grad_nonfinite")
    if np.isfinite(L1) and (L1 - L0) < -DELTA_MAX:
        reasons.append("DELTA_MAX")
    if reasons:
        LEAVES.append(dict(
            x=np.array(x1, dtype=float),
            reasons=reasons,
            dE=float(L1 - L0) if np.isfinite(L1) else None,
        ))
    return out

nuts._leapfrog = _rec_leapfrog   # nuts._build 递归内调用的也是这个 patched 版

# ---------------- 3. 薄复刻 _one_transition, 额外返回 L0 与叶节点 ----------------
def my_one_transition(x, g, lp, eps, Sinv, Lc, gf):
    """与 nuts._one_transition 逐行一致, 仅额外记录 E_start=L0 与发散叶节点。"""
    r0 = _draw_r(Lc)
    L0 = lp - 0.5 * r0 @ (Sinv @ r0)
    logu = L0 + np.log(np.random.random())
    xm = xp = x
    rm = rp = r0
    gm, gp = g, g
    lpm = lpp = lp
    j, keep = 0, True
    xcand, n_tot = x, 1
    alpha_i, nalpha_i, ndiv_i = 0.0, 0, 0
    while keep and j < J_MAX:
        v = 1 if np.random.random() < 0.5 else -1
        tr = nuts._build(xm if v == -1 else xp,
                         rm if v == -1 else rp,
                         gm if v == -1 else gp,
                         lpm if v == -1 else lpp,
                         logu, v, j, eps, Sinv, gf, L0)
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
        lp, g = gf(x)
    return x, g, lp, j - 1, alpha_i, max(nalpha_i, 1), ndiv_i, L0

# ---------------- 4. 薄复刻 _chain, 归因发散到 (chain, iter) ----------------
def my_chain(x0, n_draw, warmup, gf, target_accept, seed, C, Sinv, Lc, chain_id):
    np.random.seed(seed)
    d = x0.shape[0]
    eps = _find_reasonable_stepsize(x0, Sinv, Lc, gf)
    LEAVES.clear()

    def _cap(e, Mmat):
        try:
            lmax = float(np.linalg.eigvalsh(Mmat).max())
            return min(e, 0.9 * 2.0 / np.sqrt(max(lmax, 1e-12)))
        except Exception:
            return e
    eps = _cap(eps, C)

    gamma, t0da, kappa = 0.05, 10.0, 0.75
    mu, H_bar, log_eps_bar = np.log(10 * eps), 0.0, np.log(eps)
    m_loc = 0

    x = x0.copy()
    lp, g = gf(x)
    chain = np.zeros((n_draw, d))
    depths = np.zeros(n_draw)
    E_starts = np.zeros(n_draw)
    alpha_sum, nalpha_tot = 0.0, 0
    ndiv_w, ndiv_p = 0, 0
    div_records = []          # post-warmup 发散记录

    for i in range(n_draw):
        LEAVES.clear()
        x, g, lp, dep, ai, nai, ndi, L0 = my_one_transition(
            x, g, lp, eps, Sinv, Lc, gf)
        chain[i] = x
        depths[i] = dep
        E_starts[i] = L0
        alpha_sum += ai
        nalpha_tot += nai
        dtrans = 1 if ndi > 0 else 0
        if i < warmup:
            ndiv_w += dtrans
        else:
            ndiv_p += dtrans
            if dtrans:
                leaves_snap = [
                    dict(x=lf["x"].tolist(), reasons=lf["reasons"],
                         dE=lf["dE"]) for lf in LEAVES
                ]
                div_records.append(dict(
                    chain=chain_id, iter=i,
                    accepted_x=x.tolist(),
                    ndiv_leaves=ndi,
                    leaves=leaves_snap,
                    E_start=float(L0),
                ))

        if i < warmup:
            a_avg = ai / nai
            m_loc += 1
            H_bar = (1 - 1 / (m_loc + t0da)) * H_bar + \
                    (1 / (m_loc + t0da)) * (target_accept - a_avg)
            log_eps = mu - np.sqrt(m_loc) / gamma * H_bar
            eps = max(np.exp(log_eps), 1e-7)
            log_eps_bar = m_loc ** (-kappa) * log_eps + \
                          (1 - m_loc ** (-kappa)) * log_eps_bar
        elif i == warmup:
            eps = max(np.exp(log_eps_bar), 1e-7)

    return dict(chain=chain, depths=depths, E_starts=E_starts,
                accept=alpha_sum / max(nalpha_tot, 1),
                ndiv_w=ndiv_w, ndiv_p=ndiv_p, div_records=div_records)


def inner_min(x6, t):
    """对 6 维 x=[xq0,xe0,xb0,xq1,xe1,xb1], 返回每 cause 的最小 inner u。"""
    out = []
    for j in range(2):
        xq, xe, xb = x6[3*j:3*j+3]
        q = 2.0 - np.exp(xq)
        beta = np.exp(xb)
        lam = np.exp(-beta * xe)
        inner = 1.0 - (1.0 - q) * lam * np.power(t, beta)
        out.append(float(inner.min()))
    return out


def main():
    t0 = time.time()
    n_draw = int(os.environ.get("DIV_NDRAW", "2000"))
    warmup = int(os.environ.get("DIV_WARMUP", "1000"))
    ta = 0.9
    chains, depths, E_starts = [], [], []
    ndw = ndp = 0
    acc_list = []
    all_div_records = []
    for ci, th0 in enumerate(inits):
        th0 = np.asarray(th0, float)
        lp0, _ = gradf(th0)
        sc = 0.3
        while not np.isfinite(lp0) and sc > 1e-3:
            np.random.seed(7000 + ci)
            th0 = x_mle + sc * np.random.standard_normal(6)
            lp0, _ = gradf(th0)
            sc *= 0.5
        res = my_chain(th0, n_draw, warmup, gradf, ta, seed=900 + ci,
                       C=C0, Sinv=Sinv0, Lc=Lc0, chain_id=ci)
        chains.append(res["chain"]); depths.append(res["depths"])
        E_starts.append(res["E_starts"])
        acc_list.append(res["accept"])
        ndw += res["ndiv_w"]; ndp += res["ndiv_p"]
        all_div_records.extend(res["div_records"])
        print(f"chain {ci}: warmup_div={res['ndiv_w']} post_div={res['ndiv_p']} "
              f"accept={res['accept']:.3f}", flush=True)

    chains = np.array(chains)          # (2, 2000, 6)
    depths = np.array(depths)
    E_starts = np.array(E_starts)
    post = chains[:, warmup:, :]      # (2, 1000, 6)
    post_depths = depths[:, warmup:]

    # ---- 后验均值/协方差 (马氏距离用) ----
    flat = post.reshape(-1, 6)
    xbar = flat.mean(axis=0)
    S = np.cov(flat, rowvar=False)
    S = S + (1e-4 * np.trace(S) / 6 + 1e-8) * np.eye(6)
    Sinv_post = np.linalg.inv(S)

    # ---- 对每个 post-warmup 发散点分类 ----
    classified = []
    n_near_wall = 0
    for rec in all_div_records:
        leaves = rec["leaves"]
        # 该 transition 的"位置": 取第一个发散叶节点
        lv = leaves[0]["x"] if leaves else rec["accepted_x"]
        lv = np.asarray(lv, float)
        qE = 2.0 - np.exp(lv[0]); qD = 2.0 - np.exp(lv[3])
        d2 = float((lv - xbar) @ Sinv_post @ (lv - xbar))
        d_maha = float(np.sqrt(max(d2, 0.0)))
        xqE, xqD = lv[0], lv[3]
        wallE = bool(xqE < XQ_LO or xqE > XQ_HI)   # 软墙实际被触发
        wallD = bool(xqD < XQ_LO or xqD > XQ_HI)
        near_wallE = bool(abs(xqE - XQ_LO) < 0.15 or abs(xqE - XQ_HI) < 0.15)
        near_wallD = bool(abs(xqD - XQ_LO) < 0.15 or abs(xqD - XQ_HI) < 0.15)
        imin = inner_min(lv, t)
        barE = bool(imin[0] < 0.05)
        barD = bool(imin[1] < 0.05)
        is_near_wall = bool(wallE or wallD or near_wallE or near_wallD or barE or barD)
        if is_near_wall:
            n_near_wall += 1
        classified.append(dict(
            chain=rec["chain"], iter=rec["iter"],
            q_E=float(qE), q_D=float(qD),
            maha_dist=d_maha,
            wall_engaged_E=wallE, wall_engaged_D=wallD,
            near_wall_band_E=near_wallE, near_wall_band_D=near_wallD,
            min_inner_E=imin[0], min_inner_D=imin[1],
            near_inner_barrier_E=barE, near_inner_barrier_D=barD,
            is_near_wall_or_barrier=is_near_wall,
            leaf_reasons=[r for lf in leaves for r in lf["reasons"]],
            ndiv_leaves=rec["ndiv_leaves"],
        ))

    # ---- 汇总诊断 ----
    ess = _ess(post)
    rhat = _rhat(post)
    elapsed = time.time() - t0
    bfmi_list = []
    e_stats = []
    for c in range(2):
        Ec = E_starts[c, warmup:]
        dE = np.diff(Ec)
        bfmi = float(Ec.var(ddof=1) / dE.var(ddof=1))
        bfmi_list.append(bfmi)
        e_stats.append(dict(E_mean=float(Ec.mean()), E_var=float(Ec.var(ddof=1)),
                            dE_var=float(dE.var(ddof=1)), bfmi=bfmi))

    # 数据支持区的最小 inner (后验均值处)
    imin_mean = inner_min(xbar, t)

    out = dict(
        data=dict(n=len(df), n_E=N_E, n_D=N_D, n_censored=N_C),
        setup=dict(n_draw=n_draw, warmup=warmup, target_accept=ta,
                   n_chains=2, q_lo=Q_LO, q_hi=Q_HI,
                   xq_softwall_bounds=[float(XQ_LO), float(XQ_HI)],
                   soft_barrier=_SOFT),
        counts=dict(warmup_divergences=int(ndw),
                    post_warmup_divergences=int(ndp),
                    post_div_near_wall_or_barrier=int(n_near_wall),
                    post_div_in_high_density=int(ndp - n_near_wall)),
        tree_depth=dict(mean_post=float(post_depths.mean()),
                        max_post=float(post_depths.max())),
        accept=dict(per_chain=acc_list, mean=float(np.mean(acc_list))),
        ess_per_param=ess.tolist(),
        rhat=rhat.tolist(),
        ess_total=float(ess.sum()),
        n_grad=ngrad[0],
        elapsed_sec=elapsed,
        ess_per_sec=float(ess.sum() / elapsed),
        energy=e_stats,
        bfmi=dict(per_chain=bfmi_list, mean=float(np.mean(bfmi_list))),
        posterior_mean_x=xbar.tolist(),
        min_inner_at_posterior_mean=dict(E=imin_mean[0], D=imin_mean[1]),
        divergent_points=classified,
    )
    # accept: 从 my_chain 记录重算 (重新跑一遍太贵; 用每链 alpha 已在循环打印)
    # 这里用打印值补不上 -> 重算 accept 需要重跑; 改为在 my_chain 里返回并收集
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("divergent_points",)},
                     indent=2, default=str))
    out_fp = os.environ.get("DIV_OUT", DATA + r"\divergence_diag.json")
    json.dump(out, open(out_fp, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("WROTE", out_fp)


if __name__ == "__main__":
    main()
