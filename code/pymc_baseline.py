# -*- coding: utf-8 -*-
"""审稿人 P1-4: PyMC 标准 NUTS 基线 (Weibull q=1, 2-模式竞争风险)。
与现有 nuts.py (H&G2014 Algo6) 对比 runtime / ESS-per-sec / divergence / R-hat。
q-Weibull 非 PyMC 内建, 这里按任务要求退到 q=1 Weibull 竞争风险基线。
似然与 qweibull.cr_loglik(q=1) 一致:
  cause=0(E事件): log f_E + log S_D
  cause=1(D事件): log f_D + log S_E
  cause=-1(删失): log S_E + log S_D
输出: data/stan_baseline.json
"""
import sys, time, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
import pytensor.tensor as pt
import pymc as pm
import arviz as az

df = pd.read_csv(r"G:\OPT\q-Weibull竞争风险论文\data\voltage.csv")
t = df["hours"].to_numpy(float)
fm = df["failure_mode"].astype(str).str.lower().to_numpy()
cause = np.where(fm == "e", 0, np.where(fm == "d", 1, -1)).astype(int)
n = len(t)
print("n=", n, "E=", (cause == 0).sum(), "D=", (cause == 1).sum(), "cen=", (cause == -1).sum())

t_ = pt.constant(t.astype(float))
cause_ = pt.constant(cause.astype(float))
logt_ = pt.constant(np.log(t).astype(float))
med = max(np.median(t[cause != -1]), 1.0)

with pm.Model() as model:
    # 每模式 j: log_lam_j, log_beta_j (q=1 固定)
    log_lam = pm.Normal("log_lam", mu=np.log(1.0 / med), sigma=1.5, shape=2)
    log_beta = pm.Normal("log_beta", mu=np.log(1.5), sigma=1.2, shape=2)
    beta = pt.exp(log_beta)              # (2,)
    lam = pt.exp(log_lam)                # (2,)
    # z_ij = lam_j * t_i^beta_j ; 广播
    z = lam.dimshuffle(0, "x") * pt.power(t_, beta.dimshuffle(0, "x"))   # (2,n)
    logf = (pt.log(beta).dimshuffle(0, "x") + pt.log(lam).dimshuffle(0, "x")
            + (beta.dimshuffle(0, "x") - 1.0) * logt_ - z)               # (2,n)
    logS = -z                                                            # (2,n)
    fE, fD = logf[0], logf[1]
    sE, sD = logS[0], logS[1]
    # cause: 0=E事件,1=D事件,-1=删失
    ll = pt.switch(pt.eq(cause_, 0.0), fE + sD,
                   pt.switch(pt.eq(cause_, 1.0), fD + sE, sE + sD))
    pm.Potential("loglik", ll.sum())

    t0 = time.time()
    idata = pm.sample(draws=600, tune=400, chains=2, cores=1,
                      random_seed=2026, progressbar=False,
                      target_accept=0.90,
                      discard_tuned_samples=True)
    wall = time.time() - t0

summ = az.summary(idata, var_names=["log_lam", "log_beta"], round_to=6)
print(summ)
diag = idata.sample_stats
div = int(np.asarray(diag["diverging"]).sum())
ess_bulk = np.asarray(summ["ess_bulk"])
ess_tail = np.asarray(summ["ess_tail"])
rhat = np.asarray(summ["r_hat"])
print("wall(sampling)=%.2fs  divergences=%d  ess_bulk_sum=%.0f  rhat_max=%.3f"
      % (wall, div, ess_bulk.sum(), np.nanmax(rhat)))

out = dict(
    engine="PyMC 6.3.2 (pytensor 3.3.3, NUTS=Stan/Hoffman-Gelbaum No-U-Turn, dense mass, dual averaging)",
    model="Weibull q=1, 2-mode competing risks on voltage (n=58)",
    setup=dict(draws=600, tune=400, chains=2, target_accept=0.90, cores=1),
    wall_sampling_sec=float(wall),
    divergences=div,
    ess_bulk=[float(x) for x in ess_bulk],
    ess_tail=[float(x) for x in ess_tail],
    ess_bulk_total=float(ess_bulk.sum()),
    ess_per_sec=float(ess_bulk.sum() / wall),
    r_hat=[float(x) for x in rhat],
    r_hat_max=float(np.nanmax(rhat)),
    summary=summ.to_dict(),
    compare_to_own_nuts=dict(
        note="现有 nuts.py (q-Weibull 完整模型 A 参数化): ess_per_sec=129.5, div=18, rhat_max=1.004, wall=15.1s",
        own_nuts_ess_per_sec=129.48,
        own_nuts_div=18,
        own_nuts_rhat_max=1.004,
    ),
)
fp = r"G:\OPT\q-Weibull竞争风险论文\data\stan_baseline.json"
with open(fp, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2, default=float)
print("saved:", fp)
