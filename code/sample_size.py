# -*- coding: utf-8 -*-
"""P1-8: 样本量 n 对 q_E 估计的影响。固定 qtrue=1.45 strong。
n in {50,100,200,500}, 报告 q_E 的 MLE 与 Bayes bias/RMSE/coverage 随 n 变化。
先 R=10 smoke 验证趋势, 再放大核心 R。画 figures/fig9_n_effect.pdf+.png(300dpi)。
用法: python sample_size.py [R]   (默认 R=30; n=500 自动降 R=max(R,20))"""
import sys, os, time, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
from cr_bayes_model import default_cr_prior, cr_logpost_grad
from nuts import sample_nuts, numerical_hessian
from sim_grid import median_scale, rnd_qw, cr_ng
from scipy.optimize import minimize

ET, BT, BC = 100.0, 2.0, 3.0
WU, PD = 200, 300
QTRUE = 1.45
RHO = 0.6
NS = [50, 100, 200, 500]
# 用户定位: 趋势清楚即可, 不穷举。小 n 多重复, n=500 NUTS 慢压到 12。
RN_MAP = {50: 25, 100: 25, 200: 25, 500: 12}
if len(sys.argv) > 1:                     # 可选: 统一覆盖(做 smoke)
    _r = int(sys.argv[1]); RN_MAP = {n: _r for n in NS}
OUT = r"G:\OPT\q-Weibull竞争风险论文\data\sample_size.json"
FIG = r"G:\OPT\q-Weibull竞争风险论文\figures\fig9_n_effect"


def fit(nrep, t, cause, eta_c):
    x0 = np.concatenate([[np.log(2-QTRUE), np.log(ET**-BT), np.log(BT)],
                         [np.log(1.0), np.log(eta_c**-BC), np.log(BC)]])
    th = minimize(lambda z: cr_ng(z, t, cause), x0, jac=True, method="BFGS",
                  options=dict(maxiter=500)).x
    qm = 2-np.exp(th[0])
    xe = []
    for j in range(2):
        xq, xl, xb = th[3*j:3*j+3]; ll = np.exp(xl); bb = np.exp(xb)
        xe += [xq, np.log(ll**(-1/bb)), xb]
    xe = np.array(xe)
    priors = default_cr_prior(t, cause, q_scale=0.5)
    gradf = lambda x, pr=priors: cr_logpost_grad(x, t, cause, pr)
    dg = sample_nuts([xe], gradf, n_draw=WU+PD, warmup=WU, target_accept=0.90,
                     init_hessian=-numerical_hessian(gradf, xe, 1e-4))
    qp = 2-np.exp(dg["chains"][0, WU:, 0])
    return qm, qp, dg


def main():
    eta_c = RHO*ET*median_scale(QTRUE, BT)/median_scale(1, BC)
    res = dict(qtrue=QTRUE, rho=RHO, RN_MAP=RN_MAP, by_n={})
    for n in NS:
        Rn = RN_MAP[n]                       # 小 n 多重复, n=500 压到 12
        rng = np.random.default_rng(2026)
        mq, bq, cov, ess, div, bt = [], [], [], [], 0.0, 0.0
        t0 = time.time()
        for r in range(Rn):
            T0 = rnd_qw(QTRUE, ET, BT, n, rng)
            T1 = rnd_qw(1.0, eta_c, BC, n, rng)
            t = np.minimum(T0, T1); cause = (T1 < T0).astype(int)
            qm, qp, dg = fit(n, t, cause, eta_c)
            mq.append(qm); bq.append(np.median(qp))
            lo, hi = np.percentile(qp, [2.5, 97.5])
            cov.append(lo <= QTRUE <= hi)
            ess.append(dg["ess"][0]); div += dg["divergent"]; bt += dg["elapsed"]
        mq = np.array(mq); bq = np.array(bq)
        res["by_n"][str(n)] = dict(
            R=Rn,
            mle=dict(bias=float((mq-QTRUE).mean()), rmse=float(np.sqrt(((mq-QTRUE)**2).mean()))),
            bayes=dict(bias=float((bq-QTRUE).mean()), rmse=float(np.sqrt(((bq-QTRUE)**2).mean())),
                       coverage=float(np.mean(cov))),
            ess_q=float(np.mean(ess)), div_per_rep=float(div/Rn), sec_per_rep=float(bt/Rn))
        e = res["by_n"][str(n)]
        print(f"n={n}(R={Rn}): MLE rmse={e['mle']['rmse']:.3f} bias={e['mle']['bias']:+.3f} | "
              f"Bay rmse={e['bayes']['rmse']:.3f} bias={e['bayes']['bias']:+.3f} "
              f"cov={e['bayes']['coverage']:.2f} ess={e['ess_q']:.0f} "
              f"[{(time.time()-t0)/60:.1f}m]", flush=True)
    res["MCSE"] = {str(n): float(np.sqrt(0.95*0.05/RN_MAP[n])) for n in NS}
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # ---- fig9: RMSE / bias vs n (黑白线型) ----
    ns = sorted(int(k) for k in res["by_n"])
    mle_rmse = [res["by_n"][str(n)]["mle"]["rmse"] for n in ns]
    bay_rmse = [res["by_n"][str(n)]["bayes"]["rmse"] for n in ns]
    mle_bias = [abs(res["by_n"][str(n)]["mle"]["bias"]) for n in ns]
    bay_bias = [abs(res["by_n"][str(n)]["bayes"]["bias"]) for n in ns]
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.plot(ns, mle_rmse, "k-o", lw=1.6, ms=5, label="MLE RMSE")
    ax.plot(ns, bay_rmse, "k--s", lw=1.6, ms=5, label="Bayes RMSE")
    ax.plot(ns, mle_bias, "k:^", lw=1.4, ms=5, label="MLE |bias|")
    ax.plot(ns, bay_bias, "k-.d", lw=1.4, ms=5, label="Bayes |bias|")
    ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels(ns)
    ax.set_xlabel("sample size n"); ax.set_ylabel("error vs true q=1.45")
    ax.set_title("q_E accuracy vs sample size")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG+".pdf"); fig.savefig(FIG+".png", dpi=300)
    print("saved", OUT, "and", FIG)


if __name__ == "__main__":
    main()
