# -*- coding: utf-8 -*-
"""P1-6: q 先验中心 mu_q 敏感性。场景固定 qtrue=1.45 strong。
同一批数据集下只改 default_cr_prior 的 q_center(mu_q=0.8/1.0/1.2), sigma_q 固定 0.5。
报告 q_E 的 Bayes median bias/RMSE/coverage/后验宽度/divergence。
用法: python prior_center_sensitivity.py [R]  (默认 R=30)"""
import sys, os, time, json
import numpy as np
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
from cr_bayes_model import default_cr_prior, cr_logpost_grad
from nuts import sample_nuts, numerical_hessian
from sim_grid import median_scale, rnd_qw, cr_ng
from scipy.optimize import minimize

ET, BT, BC = 100.0, 2.0, 3.0
WU, PD = 200, 300
n = 200
R = int(sys.argv[1]) if len(sys.argv) > 1 else 30
QTRUE = 1.45
RHO = 0.6
MUQS = [0.8, 1.0, 1.2]
OUT = r"G:\OPT\q-Weibull竞争风险论文\data\prior_center_sensitivity.json"


def mle_start(t, cause):
    x0 = np.concatenate([[np.log(2-QTRUE), np.log(ET**-BT), np.log(BT)],
                         [np.log(1.0), np.log(ETA_C**-BC), np.log(BC)]])
    th = minimize(lambda z: cr_ng(z, t, cause), x0, jac=True, method="BFGS",
                  options=dict(maxiter=500)).x
    xe = []
    for j in range(2):
        xq, xl, xb = th[3*j:3*j+3]; ll = np.exp(xl); bb = np.exp(xb)
        xe += [xq, np.log(ll**(-1/bb)), xb]
    return np.array(xe)


def main():
    global ETA_C
    ETA_C = RHO*ET*median_scale(QTRUE, BT)/median_scale(1, BC)
    # 同一批数据集
    rng = np.random.default_rng(2026)
    data = []
    for r in range(R):
        T0 = rnd_qw(QTRUE, ET, BT, n, rng)
        T1 = rnd_qw(1.0, ETA_C, BC, n, rng)
        t = np.minimum(T0, T1); cause = (T1 < T0).astype(int)
        data.append((t, cause, mle_start(t, cause)))
    print(f"R={R} 数据集与 MLE 起点就绪", flush=True)

    res = dict(qtrue=QTRUE, rho=RHO, n=n, R=R, sigma_q=0.5, centers={})
    for mu in MUQS:
        meds, los, his, ess, div, bt = [], [], [], [], 0.0, 0.0
        for t, cause, xe in data:
            priors = default_cr_prior(t, cause, q_scale=0.5)
            for p in priors:
                p.mu_q = mu                      # 仅改先验中心, sigma_q=0.5
            gradf = lambda x, pr=priors: cr_logpost_grad(x, t, cause, pr)
            dg = sample_nuts([xe], gradf, n_draw=WU+PD, warmup=WU, target_accept=0.90,
                             init_hessian=-numerical_hessian(gradf, xe, 1e-4))
            qp = 2-np.exp(dg["chains"][0, WU:, 0])
            lo, hi = np.percentile(qp, [2.5, 97.5])
            meds.append(np.median(qp)); los.append(lo); his.append(hi)
            ess.append(dg["ess"][0]); div += dg["divergent"]; bt += dg["elapsed"]
        meds = np.array(meds); los = np.array(los); his = np.array(his)
        res["centers"][f"mu{mu}"] = dict(
            bias=float((meds-QTRUE).mean()), rmse=float(np.sqrt(((meds-QTRUE)**2).mean())),
            coverage=float(np.mean([l <= QTRUE <= h for l, h in zip(los, his)])),
            width=float(np.mean(his-los)), ess=float(np.mean(ess)),
            div_per_rep=float(div/R), sec_per_rep=float(bt/R))
        e = res["centers"][f"mu{mu}"]
        print(f"mu_q={mu}: bias={e['bias']:+.3f} rmse={e['rmse']:.3f} "
              f"cov={e['coverage']:.2f} width={e['width']:.3f} ess={e['ess']:.0f} "
              f"div={e['div_per_rep']:.2f}", flush=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("saved", OUT)


if __name__ == "__main__":
    main()
