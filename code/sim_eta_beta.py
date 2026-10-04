# -*- coding: utf-8 -*-
"""P1-5: qtrue x rho 6格上, 除 q_E 外, 同时报告目标模式(cause=0)的
eta_E(=100)、beta_E(=2) 的 MLE 与 Bayes posterior median 的 bias/RMSE/coverage。
qtrue=0.6(有界) 额外报 support_upper 端点估计与真值偏差。
用法: python sim_eta_beta.py [R]   (默认 R=40; 传小整数做 smoke)"""
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
R = int(sys.argv[1]) if len(sys.argv) > 1 else 40
COMBOS = [(0.6, "strong", 0.6), (0.6, "weak", 4.0),
          (1.0, "strong", 0.6), (1.0, "weak", 4.0),
          (1.45, "strong", 0.6), (1.45, "weak", 4.0)]
OUT = r"G:\OPT\q-Weibull竞争风险论文\data\sim_eta_beta.json"


def fit_one(t, cause, qtrue, eta_c):
    x0 = np.concatenate([[np.log(2-qtrue), np.log(ET**-BT), np.log(BT)],
                         [np.log(1.0), np.log(eta_c**-BC), np.log(BC)]])
    th = minimize(lambda z: cr_ng(z, t, cause), x0, jac=True, method="BFGS",
                  options=dict(maxiter=500)).x
    qm = 2-np.exp(th[0]); lam_m = np.exp(th[1]); bm = np.exp(th[2])
    em = lam_m**(-1.0/bm)                       # MLE eta = lam^(-1/beta)
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
    qp = 2-np.exp(post[:, 0]); ep = np.exp(post[:, 1]); bp = np.exp(post[:, 2])
    return dict(mle=(qm, em, bm), bay=(qp, ep, bp), dg=dg)


def stat_bias_rmse(est, truth):
    est = np.asarray(est, float)
    return dict(bias=float((est-truth).mean()), rmse=float(np.sqrt(((est-truth)**2).mean())))


def main():
    res = dict(ET=ET, BT=BT, n=n, R=R, combos={})
    for qtrue, sname, rho in COMBOS:
        rng = np.random.default_rng(2026)
        eta_c = rho*ET*median_scale(qtrue, BT)/median_scale(1, BC)
        mle_q, mle_e, mle_b = [], [], []
        bay_q, bay_e, bay_b = [], [], []
        cov_q, cov_e, cov_b = [], [], []
        ess0, div, bt = [], 0.0, 0.0
        tmax_ml, tmax_bay = [], []
        t0 = time.time()
        for r in range(R):
            T0 = rnd_qw(qtrue, ET, BT, n, rng)
            T1 = rnd_qw(1.0, eta_c, BC, n, rng)
            t = np.minimum(T0, T1); cause = (T1 < T0).astype(int)
            o = fit_one(t, cause, qtrue, eta_c)
            qm, em, bm = o["mle"]; qp, ep, bp = o["bay"]
            mle_q.append(qm); mle_e.append(em); mle_b.append(bm)
            bay_q.append(np.median(qp)); bay_e.append(np.median(ep)); bay_b.append(np.median(bp))
            lq, hq = np.percentile(qp, [2.5, 97.5])
            le, he = np.percentile(ep, [2.5, 97.5])
            lb, hb = np.percentile(bp, [2.5, 97.5])
            cov_q.append(lq <= qtrue <= hq); cov_e.append(le <= ET <= he); cov_b.append(lb <= BT <= hb)
            ess0.append(o["dg"]["ess"][0]); div += o["dg"]["divergent"]; bt += o["dg"]["elapsed"]
            if qtrue < 1.0:   # 有界: 逐 draws 算 support_upper
                tmax_ml.append(support_upper(qm, em**(-bm), bm))
                tmax_bay.append(np.median([support_upper(a, e**(-b), b) for a, e, b in zip(qp, ep, bp)]))
        mle_q = np.array(mle_q); mle_e = np.array(mle_e); mle_b = np.array(mle_b)
        bay_q = np.array(bay_q); bay_e = np.array(bay_e); bay_b = np.array(bay_b)
        ckey = f"q{qtrue}_{sname}"
        entry = dict(qtrue=qtrue, rho=rho, eta_c=float(eta_c),
                     mle=dict(q=stat_bias_rmse(mle_q, qtrue),
                              eta=stat_bias_rmse(mle_e, ET),
                              beta=stat_bias_rmse(mle_b, BT)),
                     bayes=dict(q={**stat_bias_rmse(bay_q, qtrue),
                                   "coverage": float(np.mean(cov_q))},
                                eta={**stat_bias_rmse(bay_e, ET),
                                     "coverage": float(np.mean(cov_e))},
                                beta={**stat_bias_rmse(bay_b, BT),
                                      "coverage": float(np.mean(cov_b))}),
                     ess_q=float(np.mean(ess0)), div_per_rep=float(div/R),
                     sec_per_rep=float(bt/R))
        if qtrue < 1.0:
            Tmax_true = support_upper(qtrue, ET**(-BT), BT)
            tmax_ml = np.array(tmax_ml); tmax_bay = np.array(tmax_bay)
            entry["Tmax"] = dict(true=float(Tmax_true),
                                 mle=stat_bias_rmse(tmax_ml, Tmax_true),
                                 bayes=stat_bias_rmse(tmax_bay, Tmax_true))
        res["combos"][ckey] = entry
        print(f"{ckey}: MLE q={entry['mle']['q']['bias']:+.3f}/{entry['mle']['q']['rmse']:.3f} "
              f"eta={entry['mle']['eta']['bias']:+.1f}/{entry['mle']['eta']['rmse']:.1f} "
              f"beta={entry['mle']['beta']['bias']:+.3f}/{entry['mle']['beta']['rmse']:.3f} | "
              f"Bay q cov={entry['bayes']['q']['coverage']:.2f} eta cov={entry['bayes']['eta']['coverage']:.2f} "
              f"beta cov={entry['bayes']['beta']['coverage']:.2f} "
              f"[{(time.time()-t0)/60:.1f}m]", flush=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("saved", OUT)


if __name__ == "__main__":
    main()
