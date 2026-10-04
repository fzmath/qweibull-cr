# -*- coding: utf-8 -*-
"""必改2: q 先验敏感性。
Part A 仿真 q=1.45 strong: N(1,.25)/.5/1.0 与 U(-1,2), 在相同 R 数据集上对比。
Part B 真实 voltage: 4 先验各一次, 看 q_E/q_D 后验稳健性。"""
import sys, os, json
import numpy as np
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
import pandas as pd
from qweibull import Q_MIN
from cr_bayes_model import default_cr_prior, cr_logpost_grad
from nuts import sample_nuts, numerical_hessian
from sim_grid import rnd_qw, median_scale, cr_ng
from scipy.optimize import minimize

CONFIGS = [("N(1,0.25)", False, 1.0, 0.25, Q_MIN, 1.98),
           ("N(1,0.5)", False, 1.0, 0.50, Q_MIN, 1.98),
           ("N(1,1.0)", False, 1.0, 1.00, Q_MIN, 1.98),
           ("U(-1,2)", True, None, None, -1.0, 2.0)]


def set_q_prior(priors, cfg):
    name, uniform, mu, sig, q_lo, q_hi = cfg
    for p in priors:
        p.uniform = uniform
        p.q_lo, p.q_hi = q_lo, q_hi
        if not uniform:
            p.mu_q, p.sigma_q = mu, sig


def mle_start(t, cause, qtrue, eta_c):
    x0 = np.concatenate([[np.log(2-qtrue), np.log(100.0**-2.0), np.log(2.0)],
                         [np.log(1), np.log(eta_c**-3.0), np.log(3.0)]])
    th = minimize(lambda z: cr_ng(z, t, cause), x0, jac=True, method="BFGS",
                  options=dict(maxiter=500)).x
    xe = []
    for j in range(2):
        xq, xl, xb = th[3*j:3*j+3]; ll = np.exp(xl); bb = np.exp(xb)
        xe += [xq, np.log(ll**(-1/bb)), xb]
    return np.array(xe)


# ---------- Part A: 仿真 ----------
qtrue, R = 1.45, int(os.environ.get("PSIM_R", "40"))
eta_c = 0.6*100.0*median_scale(qtrue, 2.0)/median_scale(1, 3.0)
rng = np.random.default_rng(2026)
data = []
for r in range(R):
    T0 = rnd_qw(qtrue, 100.0, 2.0, 200, rng)
    T1 = rnd_qw(1.0, eta_c, 3.0, 200, rng)
    t = np.minimum(T0, T1); cause = (T1 < T0).astype(int)
    data.append((t, cause, mle_start(t, cause, qtrue, eta_c)))
print(f"Part A: {R} 数据集与 MLE 起点就绪", flush=True)

partA = {}
for cfg in CONFIGS:
    meds, los, his, sds, ess, div = [], [], [], [], [], 0.0
    for t, cause, xe in data:
        priors = default_cr_prior(t, cause); set_q_prior(priors, cfg)
        gradf = lambda x, pr=priors: cr_logpost_grad(x, t, cause, pr)
        dg = sample_nuts([xe], gradf, n_draw=500, warmup=200, target_accept=0.9,
                         init_hessian=-numerical_hessian(gradf, xe, 1e-4))
        qp = 2-np.exp(dg["chains"][0, 200:, 0])
        lo, hi = np.percentile(qp, [2.5, 97.5])
        meds.append(np.median(qp)); los.append(lo); his.append(hi)
        sds.append(qp.std(ddof=1)); ess.append(dg["ess"][0]); div += dg["divergent"]
    meds = np.array(meds); los = np.array(los); his = np.array(his)
    partA[cfg[0]] = dict(bias=float((meds-qtrue).mean()),
                         rmse=float(np.sqrt(((meds-qtrue)**2).mean())),
                         coverage=float(np.mean([l <= qtrue <= h
                                                 for l, h in zip(los, his)])),
                         width=float(np.mean(his-los)),
                         post_sd=float(np.mean(sds)), ess=float(np.mean(ess)),
                         div_per_rep=float(div/R))
    print(f"{cfg[0]}: bias={partA[cfg[0]]['bias']:.3f} "
          f"rmse={partA[cfg[0]]['rmse']:.3f} cov={partA[cfg[0]]['coverage']:.2f} "
          f"width={partA[cfg[0]]['width']:.2f} ess={partA[cfg[0]]['ess']:.0f} "
          f"div={partA[cfg[0]]['div_per_rep']:.2f}", flush=True)

# ---------- Part B: 真实数据 ----------
df = pd.read_csv(r"G:\OPT\q-Weibull竞争风险论文\data\voltage.csv")
tv = df["hours"].to_numpy(float)
fm = df["failure_mode"].to_numpy(str)
cv = np.full(len(df), -1); cv[fm == "E"] = 0; cv[fm == "D"] = 1
xe_v = mle_start(tv, cv, 1.45, 100.0)
partB = {}
for cfg in CONFIGS:
    priors = default_cr_prior(tv, cv); set_q_prior(priors, cfg)
    gradf = lambda x, pr=priors: cr_logpost_grad(x, tv, cv, pr)
    dg = sample_nuts([xe_v], gradf, n_draw=2000, warmup=1000, target_accept=0.9,
                     init_hessian=-numerical_hessian(gradf, xe_v, 1e-4))
    post = dg["chains"][0, 1000:, :]
    out = {}
    for j, lab in [(0, "E"), (1, "D")]:
        qp = 2-np.exp(post[:, 3*j])
        out[f"q_{lab}"] = dict(median=float(np.median(qp)),
                               lo=float(np.percentile(qp, 2.5)),
                               hi=float(np.percentile(qp, 97.5)),
                               P_gt1=float((qp > 1).mean()))
    out["divergent"] = int(dg["divergent"])
    partB[cfg[0]] = out
    print(f"voltage {cfg[0]}: q_E={out['q_E']['median']:.2f}"
          f"[{out['q_E']['lo']:.2f},{out['q_E']['hi']:.2f}] P>1={out['q_E']['P_gt1']:.2f}"
          f" | q_D={out['q_D']['median']:.2f}[{out['q_D']['lo']:.2f},"
          f"{out['q_D']['hi']:.2f}] div={out['divergent']}", flush=True)

json.dump(dict(partA=partA, partB=partB, R=R),
          open(r"G:\OPT\q-Weibull竞争风险论文\data\prior_sensitivity.json", "w",
               encoding="utf-8"), ensure_ascii=False, indent=2)
print("saved prior_sensitivity.json")
