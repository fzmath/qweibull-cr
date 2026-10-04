# -*- coding: utf-8 -*-
"""必改4: smooth barrier (support wall) 阈值 epsilon_s 敏感性。
仿真 q=0.6 strong (bounded, wall 相关), eps in 1e-2..1e-5 在相同 R 数据上对比
q 的 bias/RMSE/coverage/width/ESS/发散; 并诊断观测区最小 inner, 验证 barrier
在数据支撑区不激活 (改动严格为 0, 而非"1e-18")。"""
import sys, os, json
import numpy as np
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
import bayes_model
from cr_bayes_model import default_cr_prior, cr_logpost_grad
from nuts import sample_nuts, numerical_hessian
from sim_grid import rnd_qw, median_scale, cr_ng
from scipy.optimize import minimize

EPS_LIST = [1e-2, 1e-3, 1e-4, 1e-5]
qtrue, R = 0.6, int(os.environ.get("BSIM_R", "40"))
eta_c = 0.6*100.0*median_scale(qtrue, 2.0)/median_scale(1, 3.0)
rng = np.random.default_rng(2026)
data = []
min_inner = []
for r in range(R):
    T0 = rnd_qw(qtrue, 100.0, 2.0, 200, rng)
    T1 = rnd_qw(1.0, eta_c, 3.0, 200, rng)
    t = np.minimum(T0, T1); cause = (T1 < T0).astype(int)
    x0 = np.concatenate([[np.log(2-qtrue), np.log(100.0**-2.0), np.log(2.0)],
                         [np.log(1), np.log(eta_c**-3.0), np.log(3.0)]])
    th = minimize(lambda z: cr_ng(z, t, cause), x0, jac=True, method="BFGS",
                  options=dict(maxiter=500)).x
    # 目标模式 MLE 参数下观测点 inner
    qh = 2-np.exp(th[0]); ll = np.exp(th[1]); bb = np.exp(th[2])
    inner = 1-(1-qh)*ll*t**bb
    min_inner.append(float(inner.min()))
    xe = []
    for j in range(2):
        xq, xl, xb = th[3*j:3*j+3]; l2 = np.exp(xl); b2 = np.exp(xb)
        xe += [xq, np.log(l2**(-1/b2)), xb]
    data.append((t, cause, np.array(xe)))
min_inner = np.array(min_inner)
print(f"观测区最小 inner: median={np.median(min_inner):.4f} "
      f"p5={np.percentile(min_inner,5):.4f} min={min_inner.min():.5f}", flush=True)
print("(若 min inner > 1e-2, 则 eps<=1e-2 时 barrier 在观测区均不激活)", flush=True)

res = {}
for eps in EPS_LIST:
    bayes_model._SOFT = eps
    meds, los, his, sds, ess, div = [], [], [], [], [], 0.0
    for t, cause, xe in data:
        priors = default_cr_prior(t, cause)
        gradf = lambda x, pr=priors: cr_logpost_grad(x, t, cause, pr)
        dg = sample_nuts([xe], gradf, n_draw=500, warmup=200, target_accept=0.9,
                         init_hessian=-numerical_hessian(gradf, xe, 1e-4))
        qp = 2-np.exp(dg["chains"][0, 200:, 0])
        lo, hi = np.percentile(qp, [2.5, 97.5])
        meds.append(np.median(qp)); los.append(lo); his.append(hi)
        sds.append(qp.std(ddof=1)); ess.append(dg["ess"][0]); div += dg["divergent"]
    meds = np.array(meds); los = np.array(los); his = np.array(his)
    key = f"{eps:.0e}"
    res[key] = dict(bias=float((meds-qtrue).mean()),
                    rmse=float(np.sqrt(((meds-qtrue)**2).mean())),
                    coverage=float(np.mean([l <= qtrue <= h
                                            for l, h in zip(los, his)])),
                    width=float(np.mean(his-los)), post_sd=float(np.mean(sds)),
                    ess=float(np.mean(ess)), div_per_rep=float(div/R))
    print(f"eps={key}: bias={res[key]['bias']:.3f} rmse={res[key]['rmse']:.3f} "
          f"cov={res[key]['coverage']:.2f} width={res[key]['width']:.2f} "
          f"ess={res[key]['ess']:.0f} div={res[key]['div_per_rep']:.2f}", flush=True)

bayes_model._SOFT = 1e-3
json.dump(dict(res=res, R=R, min_inner=dict(
    median=float(np.median(min_inner)), p5=float(np.percentile(min_inner, 5)),
    min=float(min_inner.min()))),
    open(r"G:\OPT\q-Weibull竞争风险论文\data\barrier_sensitivity.json", "w",
         encoding="utf-8"), ensure_ascii=False, indent=2)
print("saved barrier_sensitivity.json")
