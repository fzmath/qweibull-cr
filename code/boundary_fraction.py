# -*- coding: utf-8 -*-
"""P1-7: 边界截断比 max(t_i)/T_max。有界情形 qtrue=0.6(support_upper 有限)。
生成完整 T0 后, 按 f in {0.5,0.7,0.85,0.95,0.99} 取观测上限 tau=f*T_max(true),
t>tau 视为右删失(cause=-1, 与 prior_sensitivity Part B 删失编码一致), t_obs=min(t,tau)。
报告各 f 下 q_E/beta_E/eta_E 的 MLE 与 Bayes 估计、divergence、ESS、coverage。
用法: python boundary_fraction.py [R]  (默认 R=20)"""
import sys, os, time, json
import numpy as np
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
from qweibull import support_upper
from cr_bayes_model import default_cr_prior, cr_logpost_grad
from nuts import sample_nuts, numerical_hessian
from sim_grid import median_scale, rnd_qw, cr_ng
from scipy.optimize import minimize

ET, BT, BC = 100.0, 2.0, 3.0
WU, PD = 200, 300
n = 200
R = int(sys.argv[1]) if len(sys.argv) > 1 else 20
QTRUE = 0.6
RHO = 4.0            # weak: 竞争事件晚, 目标 q-W 事件主导, max(t) 才逼近 T_max(边界截断才咬得到)
FRACS = [0.5, 0.7, 0.85, 0.95, 0.99]
OUT = r"G:\OPT\q-Weibull竞争风险论文\data\boundary_fraction.json"


def fit(t, cause):
    x0 = np.concatenate([[np.log(2-QTRUE), np.log(ET**-BT), np.log(BT)],
                         [np.log(1.0), np.log(ETA_C**-BC), np.log(BC)]])
    th = minimize(lambda z: cr_ng(z, t, cause), x0, jac=True, method="BFGS",
                  options=dict(maxiter=500)).x
    qm = 2-np.exp(th[0]); lam_m = np.exp(th[1]); bm = np.exp(th[2]); em = lam_m**(-1.0/bm)
    xe = []
    for j in range(2):
        xq, xl, xb = th[3*j:3*j+3]; ll = np.exp(xl); bb = np.exp(xb)
        xe += [xq, np.log(ll**(-1/bb)), xb]
    xe = np.array(xe)
    priors = default_cr_prior(t, cause, q_scale=0.5)
    gradf = lambda x, pr=priors: cr_logpost_grad(x, t, cause, pr)
    dg = sample_nuts([xe], gradf, n_draw=WU+PD, warmup=WU, target_accept=0.90,
                     init_hessian=-numerical_hessian(gradf, xe, 1e-4))
    post = dg["chains"][0, WU:, :]
    return dict(mle=(qm, em, bm), bay=(2-np.exp(post[:, 0]), np.exp(post[:, 1]), np.exp(post[:, 2])), dg=dg)


def bm(est, truth):
    est = np.asarray(est, float)
    return dict(bias=float((est-truth).mean()), rmse=float(np.sqrt(((est-truth)**2).mean())))


def main():
    global ETA_C
    ETA_C = RHO*ET*median_scale(QTRUE, BT)/median_scale(1, BC)
    Tmax_true = support_upper(QTRUE, ET**(-BT), BT)
    rng = np.random.default_rng(2026)
    # 完整数据(未截断)一次性生成
    full = []
    for r in range(R):
        T0 = rnd_qw(QTRUE, ET, BT, n, rng)
        T1 = rnd_qw(1.0, ETA_C, BC, n, rng)
        t = np.minimum(T0, T1); cause = (T1 < T0).astype(int)
        full.append((t, cause))
    print(f"R={R} 完整数据集就绪, Tmax(true)={Tmax_true:.2f}", flush=True)

    res = dict(qtrue=QTRUE, rho=RHO, n=n, R=R, Tmax_true=float(Tmax_true), fractions={})
    for f in FRACS:
        tau = f*Tmax_true
        mq, me, mb, bq, be, bb = [], [], [], [], [], []
        cq, ce, cb, ess, div, bt = [], [], [], [], 0.0, 0.0
        tcens = []
        for t, cause in full:
            mask = t > tau
            t_obs = np.minimum(t, tau)
            c_obs = cause.copy(); c_obs[mask] = -1      # 右删失
            tcens.append(mask.mean())
            o = fit(t_obs, c_obs)
            qm, em, bm_ = o["mle"]; qp, ep, bp = o["bay"]
            mq.append(qm); me.append(em); mb.append(bm_)
            bq.append(np.median(qp)); be.append(np.median(ep)); bb.append(np.median(bp))
            cq.append(np.percentile(qp, 2.5) <= QTRUE <= np.percentile(qp, 97.5))
            ce.append(np.percentile(ep, 2.5) <= ET <= np.percentile(ep, 97.5))
            cb.append(np.percentile(bp, 2.5) <= BT <= np.percentile(bp, 97.5))
            ess.append(o["dg"]["ess"][0]); div += o["dg"]["divergent"]; bt += o["dg"]["elapsed"]
        mq = np.array(mq); me = np.array(me); mb = np.array(mb)
        bq = np.array(bq); be = np.array(be); bb = np.array(bb)
        key = f"f{f}"
        res["fractions"][key] = dict(
            tau=float(tau), censoring_rate=float(np.mean(tcens)),
            mle=dict(q=bm(mq, QTRUE), eta=bm(me, ET), beta=bm(mb, BT)),
            bayes=dict(q={**bm(bq, QTRUE), "coverage": float(np.mean(cq))},
                       eta={**bm(be, ET), "coverage": float(np.mean(ce))},
                       beta={**bm(bb, BT), "coverage": float(np.mean(cb))}),
            ess_q=float(np.mean(ess)), div_per_rep=float(div/R), sec_per_rep=float(bt/R))
        e = res["fractions"][key]
        print(f"{key}: tau={tau:.1f} cens={e['censoring_rate']:.2f} | "
              f"MLE q={e['mle']['q']['bias']:+.3f}/{e['mle']['q']['rmse']:.3f} "
              f"eta={e['mle']['eta']['bias']:+.1f} beta={e['mle']['beta']['bias']:+.3f} | "
              f"Bay q cov={e['bayes']['q']['coverage']:.2f} ess={e['ess_q']:.0f} div={e['div_per_rep']:.2f}",
              flush=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("saved", OUT)


if __name__ == "__main__":
    main()
