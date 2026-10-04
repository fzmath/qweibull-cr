# -*- coding: utf-8 -*-
"""用精确 MLE 起点重跑 prior sensitivity Part B (voltage), 更新 json。"""
import json
import numpy as np, pandas as pd
from cr_bayes_model import default_cr_prior, cr_logpost_grad
from nuts import sample_nuts, numerical_hessian

CONFIGS = [("N(1,0.25)", False, 1.0, 0.25, 0.20, 1.98),
           ("N(1,0.5)", False, 1.0, 0.50, 0.20, 1.98),
           ("N(1,1.0)", False, 1.0, 1.00, 0.20, 1.98),
           ("U(-1,2)", True, None, None, -1.0, 2.0)]


def set_q(priors, cfg):
    _, uniform, mu, sig, lo, hi = cfg
    for p in priors:
        p.uniform = uniform; p.q_lo, p.q_hi = lo, hi
        if not uniform:
            p.mu_q, p.sigma_q = mu, sig


df = pd.read_csv(r"G:\OPT\q-Weibull竞争风险论文\data\voltage.csv")
t = df["hours"].to_numpy(float)
fm = df["failure_mode"].to_numpy(str)
cause = np.full(len(df), -1); cause[fm == "E"] = 0; cause[fm == "D"] = 1
mle = {0: dict(q=1.2414715, lam=0.01414739, beta=0.6550399),
       1: dict(q=0.9345138, lam=1.5741296e-14, beta=5.4219912)}
xe = []
for j in range(2):
    m = mle[j]; eta = m["lam"]**(-1/m["beta"])
    xe += [np.log(2-m["q"]), np.log(eta), np.log(m["beta"])]
xe = np.array(xe)

partB = {}
for cfg in CONFIGS:
    priors = default_cr_prior(t, cause); set_q(priors, cfg)
    gradf = lambda x, pr=priors: cr_logpost_grad(x, t, cause, pr)
    rng = np.random.RandomState(3)
    dg = sample_nuts([xe, xe+0.1*rng.standard_normal(6)], gradf, n_draw=2000,
                     warmup=1000, target_accept=0.9,
                     init_hessian=-numerical_hessian(gradf, xe, 1e-4))
    post = dg["chains"][:, 1000:, :]
    out = {}
    for j, lab in [(0, "E"), (1, "D")]:
        qp = 2-np.exp(post[:, :, 3*j])
        out[f"q_{lab}"] = dict(median=float(np.median(qp)),
                               lo=float(np.percentile(qp, 2.5)),
                               hi=float(np.percentile(qp, 97.5)),
                               P_gt1=float((qp > 1).mean()))
    out["divergent"] = int(dg["divergent"])
    partB[cfg[0]] = out
    print(f"{cfg[0]}: q_E={out['q_E']['median']:.2f}[{out['q_E']['lo']:.2f},"
          f"{out['q_E']['hi']:.2f}] P>1={out['q_E']['P_gt1']:.2f} | "
          f"q_D={out['q_D']['median']:.2f}[{out['q_D']['lo']:.2f},"
          f"{out['q_D']['hi']:.2f}] div={out['divergent']}")

fp = r"G:\OPT\q-Weibull竞争风险论文\data\prior_sensitivity.json"
d = json.load(open(fp, encoding="utf-8"))
d["partB"] = partB
json.dump(d, open(fp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("updated prior_sensitivity.json partB")
