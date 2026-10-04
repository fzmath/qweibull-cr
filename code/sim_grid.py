# -*- coding: utf-8 -*-
"""必改1+必改3(仿真): 多 q 值 × 竞争截断(ρ比值) 全量仿真 R=80。
估计: MLE q-W CR, Bayes q-W CR (NUTS), MLE Weibull CR (q固定1, 误设基线)。
针对目标模式 q 报 Bias/RMSE/MAE/coverage/区间宽/后验SD/ESS/发散/耗时。
每(q,ρ)组合存独立 JSON, 已存在则跳过(断点续跑)。
用法: python sim_grid.py            # 跑全部
      python sim_grid.py 0.6 1.45   # 只跑指定 q 值"""
import sys, os, time, json
import numpy as np
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
from qweibull import Q_MIN, Q_MAX
from bayes_model import _loglik_and_grad_raw
from cr_bayes_model import default_cr_prior, cr_logpost_grad
from nuts import sample_nuts, numerical_hessian
from scipy.optimize import minimize

Q_GRID = [0.6, 0.8, 1.0, 1.25, 1.45, 1.7]
ET, BT = 100.0, 2.0          # 目标 q-W 尺度/形状
BC = 3.0                     # 竞争 Weibull 形状
RHOS = [("strong", 0.6), ("moderate", 1.0), ("weak", 4.0)]
OUTDIR = r"G:\OPT\q-Weibull竞争风险论文\data\sim_grid"
os.makedirs(OUTDIR, exist_ok=True)
n, R, WU, PD = 200, int(os.environ.get("SIM_R", "80")), 200, 300


def median_scale(q, beta):
    """t_median / eta (尺度族, 与 eta 无关)。"""
    if abs(q - 1) < 1e-9:
        return np.log(2) ** (1 / beta)
    return ((1 - 0.5 ** ((1 - q) / (2 - q))) / (1 - q)) ** (1 / beta)


def rnd_qw(q, eta, beta, m, rng):
    u = np.maximum(rng.random(m), 1e-12); lam = eta ** -beta
    if abs(q - 1) < 1e-9:
        return eta * (-np.log(u)) ** (1 / beta)
    d = 1 - q
    return ((1 - u ** (1 / ((2 - q) / d))) / (d * lam)) ** (1 / beta)


def cr_ng(theta, t, cause):            # q-W MLE (6维)
    neg = 0; g = np.zeros(6)
    for j in range(2):
        xq, xl, xb = theta[3*j:3*j+3]; q = 2-np.exp(xq); lam = np.exp(xl); beta = np.exp(xb)
        if q < Q_MIN or q > Q_MAX: return 1e10, np.zeros(6)
        ll, gq, gl, gb = _loglik_and_grad_raw(q, lam, beta, t, (cause == j).astype(int))
        if not np.isfinite(ll): return 1e10, np.zeros(6)
        neg += ll; g[3*j] = gq*-np.exp(xq); g[3*j+1] = gl*lam; g[3*j+2] = gb*beta
    return -neg, -g


def cr_ng_w(theta, t, cause):          # Weibull MLE (4维, q固定1)
    neg = 0; g = np.zeros(4)
    for j in range(2):
        xl, xb = theta[2*j:2*j+2]; lam = np.exp(xl); beta = np.exp(xb)
        ll, gq, gl, gb = _loglik_and_grad_raw(1.0, lam, beta, t, (cause == j).astype(int))
        if not np.isfinite(ll): return 1e10, np.zeros(4)
        neg += ll; g[2*j] = gl*lam; g[2*j+1] = gb*beta
    return -neg, -g


def one_combo(qtrue, rho):
    rng = np.random.default_rng(2026)
    tmed_t = ET*median_scale(qtrue, BT)
    eta_c = rho*tmed_t/median_scale(1, BC)
    mle, wald = [], []
    bmed, blo, bhi, bsd, bess, bdiv, btime = [], [], [], [], [], [], 0.0
    for r in range(R):
        T0 = rnd_qw(qtrue, ET, BT, n, rng)
        T1 = rnd_qw(1.0, eta_c, BC, n, rng)
        t = np.minimum(T0, T1); cause = (T1 < T0).astype(int)
        # MLE q-W
        x0 = np.concatenate([[np.log(2-qtrue), np.log(ET**-BT), np.log(BT)],
                             [np.log(1), np.log(eta_c**-BC), np.log(BC)]])
        th = minimize(lambda z: cr_ng(z, t, cause), x0, jac=True, method="BFGS",
                      options=dict(maxiter=500)).x
        qm = 2-np.exp(th[0]); mle.append(qm)
        H = numerical_hessian(lambda z: cr_ng(z, t, cause), th, 1e-3)
        try:
            Ci = np.linalg.inv(H+1e-6*np.eye(6)); se = np.exp(th[0])*np.sqrt(max(Ci[0, 0], 0))
            se = min(se, 5); wald.append(qm-1.96*se <= qtrue <= qm+1.96*se)
        except Exception:
            wald.append(False)
        # MLE Weibull (q=1) — 只在最后聚合需要, 其 q 点估计恒为 1
        # Bayes q-W
        priors = default_cr_prior(t, cause)
        gradf = lambda x: cr_logpost_grad(x, t, cause, priors)
        xe = []
        for j in range(2):
            xq, xl, xb = th[3*j:3*j+3]; ll = np.exp(xl); bb = np.exp(xb)
            xe += [xq, np.log(ll**(-1/bb)), xb]
        dg = sample_nuts([np.array(xe)], gradf, n_draw=WU+PD, warmup=WU, target_accept=0.90,
                         init_hessian=-numerical_hessian(gradf, np.array(xe), 1e-4))
        qp = 2-np.exp(dg["chains"][0, WU:, 0])
        md = np.median(qp); lo, hi = np.percentile(qp, [2.5, 97.5])
        bmed.append(md); blo.append(lo); bhi.append(hi); bsd.append(qp.std(ddof=1))
        bess.append(dg["ess"][0]); bdiv.append(dg["divergent"]); btime += dg["elapsed"]
    mle = np.array(mle); bmed = np.array(bmed)
    res = dict(qtrue=qtrue, rho=rho, eta_c=float(eta_c), n=n, R=R,
               mle=dict(bias=float((mle-qtrue).mean()), rmse=float(np.sqrt(((mle-qtrue)**2).mean())),
                        mae=float(np.abs(mle-qtrue).mean()), coverage=float(np.mean(wald))),
               bayes=dict(bias=float((bmed-qtrue).mean()), rmse=float(np.sqrt(((bmed-qtrue)**2).mean())),
                          mae=float(np.abs(bmed-qtrue).mean()),
                          coverage=float(np.mean([l <= qtrue <= h for l, h in zip(blo, bhi)])),
                          width=float(np.mean(np.array(bhi)-np.array(blo))),
                          post_sd=float(np.mean(bsd)), ess=float(np.mean(bess)),
                          divergences=float(np.mean(bdiv)), time_per_rep=float(btime/R)),
               weibull=dict(bias=float(1-qtrue), rmse=float(abs(1-qtrue)),
                            coverage=float(qtrue == 1.0)))
    return res


if __name__ == "__main__":
    todo = [float(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else Q_GRID
    for qtrue in Q_GRID:
        if qtrue not in todo:
            continue
        for sname, rho in RHOS:
            fp = os.path.join(OUTDIR, f"sim_q{qtrue}_{sname}.json")
            if os.path.exists(fp):
                try:
                    old = json.load(open(fp, encoding="utf-8"))
                    if old.get("R") == R:
                        print(f"skip q={qtrue} {sname} (R={R} exists)", flush=True)
                        continue
                    print(f"redo q={qtrue} {sname} (old R={old.get('R')} != {R})",
                          flush=True)
                except Exception:
                    pass  # 文件损坏, 重新计算
            t = time.time()
            res = one_combo(qtrue, rho)
            json.dump(res, open(fp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            print(f"q={qtrue} {sname}(rho={rho}): MLE rmse={res['mle']['rmse']:.3f} "
                  f"cov={res['mle']['coverage']:.2f} | Bayes rmse={res['bayes']['rmse']:.3f} "
                  f"cov={res['bayes']['coverage']:.2f} width={res['bayes']['width']:.2f} "
                  f"ess={res['bayes']['ess']:.0f} div={res['bayes']['divergences']:.2f} "
                  f"[{(time.time()-t)/60:.1f}m]", flush=True)
    print("ALL DONE")
