# -*- coding: utf-8 -*-
"""对 voltage (电枢条段) 数据拟合竞争风险模型并做模型比较 (MLE)。"""
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
import qweibull as qw

K = 2  # 模式1=E 早期, 模式2=D 退化

df = pd.read_csv(r"G:\OPT\q-Weibull竞争风险论文\data\voltage.csv")
df["cause"] = df["failure_mode"].map({"E": 1, "D": 2, "censored": 0})
t = df["hours"].to_numpy(float)
cause = df["cause"].to_numpy(int)
n = len(df)
print(f"样本 n={n}; E(模式1)={np.sum(cause==1)}, D(模式2)={np.sum(cause==2)}, 删失={np.sum(cause==0)}")

lo_q, hi_q = qw.xq_bounds()
BQ = [(lo_q, hi_q)] * K + [(None, None)] * (2 * K)


def fit_qweibull_cr():
    best = None
    t1, t2 = t[cause == 1], t[cause == 2]
    means = [max(t1.mean(), 1e-9), max(t2.mean(), 1e-9)]
    for q0 in [0.7, 0.9, 1.0, 1.1, 1.3]:
        for b0 in [1.0, 2.0, 3.0, 4.0]:
            q = np.array([q0, q0])
            beta = np.array([b0, b0])
            lam = np.array([1.0 / means[0] ** b0, 1.0 / means[1] ** b0])
            theta0 = qw.pack(q, lam, beta)
            r = minimize(qw.cr_neg_loglik, theta0, args=(K, t, cause, False),
                         method="L-BFGS-B", bounds=BQ,
                         options={"gtol": 1e-8, "maxiter": 2000})
            if best is None or r.fun < best.fun:
                best = r
    q, lam, beta = qw.unpack(best.x, K)
    return q, lam, beta, -best.fun, best


def fit_weibull_cr():
    best = None
    t1, t2 = t[cause == 1], t[cause == 2]
    means = [max(t1.mean(), 1e-9), max(t2.mean(), 1e-9)]
    for b0 in [0.8, 1.2, 2.0, 3.0, 5.0]:
        beta = np.array([b0, b0])
        lam = np.array([1.0 / means[0] ** b0, 1.0 / means[1] ** b0])
        theta0 = np.concatenate([np.log(lam), np.log(beta)])
        r = minimize(qw.cr_neg_loglik, theta0, args=(K, t, cause, True),
                     method="L-BFGS-B", bounds=[(None, None)] * (2 * K),
                     options={"gtol": 1e-8})
        if best is None or r.fun < best.fun:
            best = r
    lam = np.exp(best.x[0:K])
    beta = np.exp(best.x[K:2 * K])
    return np.ones(K), lam, beta, -best.fun, best


def fit_single_weibull():
    event = (cause > 0).astype(int)

    def nll(theta):
        lam, beta = np.exp(theta[0]), np.exp(theta[1])
        v = 0.0
        for i in range(n):
            v -= qw.log_pdf(1.0, lam, beta, t[i])[()] if event[i] else qw.log_sf(1.0, lam, beta, t[i])[()]
        return v if np.isfinite(v) else 1e10

    best = None
    for b0 in [0.8, 1.5, 2.5, 4.0]:
        lam0 = 1.0 / max(t[event == 1].mean(), 1e-9) ** b0
        r = minimize(nll, np.array([np.log(lam0), np.log(b0)]),
                     method="L-BFGS-B", bounds=[(None, None)] * 2)
        if best is None or r.fun < best.fun:
            best = r
    return np.array([1.0]), np.array([np.exp(best.x[0])]), np.array([np.exp(best.x[1])]), -best.fun, best


print("\n===== q-Weibull 竞争风险 =====")
q_q, l_q, b_q, ll_q, _ = fit_qweibull_cr()
for j, nm in enumerate(["E 早期", "D 退化"]):
    print(f"  模式{j+1} ({nm}): q={q_q[j]:.4f}, beta={b_q[j]:.4f}, lambda={l_q[j]:.3e}, 支撑上界={qw.support_upper(q_q[j], l_q[j], b_q[j]):.1f}")
print(f"  logL={ll_q:.4f}")

print("\n===== 标准 Weibull 竞争风险 (q=1) =====")
q_w, l_w, b_w, ll_w, _ = fit_weibull_cr()
for j, nm in enumerate(["E 早期", "D 退化"]):
    print(f"  模式{j+1} ({nm}): beta={b_w[j]:.4f}, eta={l_w[j]**(-1/b_w[j]):.2f}")
print(f"  logL={ll_w:.4f}")

print("\n===== 单一 Weibull (不分模式) =====")
_, l_s, b_s, ll_s, _ = fit_single_weibull()
print(f"  beta={b_s[0]:.4f}, eta={l_s[0]**(-1/b_s[0]):.2f}, logL={ll_s:.4f}")

print("\n===== 模型比较 =====")
rows = []
for nm, ll, p in [("Single Weibull", ll_s, 2), ("Weibull CR", ll_w, 4), ("q-Weibull CR", ll_q, 6)]:
    aic, bic = 2 * p - 2 * ll, p * np.log(n) - 2 * ll
    rows.append((nm, p, ll, aic, bic))
    print(f"  {nm:16s} p={p} logL={ll:9.4f} AIC={aic:9.3f} BIC={bic:9.3f}")

lrt, pval = 2 * (ll_q - ll_w), stats.chi2.sf(2 * (ll_q - ll_w), 2)
print(f"\n  LRT q-Weibull CR vs Weibull CR: 2dLL={lrt:.4f}, df=2, p={pval:.5f}")
print(f"  LRT Weibull CR vs Single: 2dLL={2*(ll_w-ll_s):.4f}, df=2, p={stats.chi2.sf(2*(ll_w-ll_s),2):.5g}")

pd.DataFrame(rows, columns=["model", "n_params", "logL", "AIC", "BIC"]).to_csv(
    r"G:\OPT\q-Weibull竞争风险论文\data\model_comparison.csv", index=False, encoding="utf-8-sig")
pd.DataFrame({"mode": ["E_early", "D_degrad"], "q": q_q, "beta_qw": b_q,
              "lambda_qw": l_q, "beta_w": b_w, "lambda_w": l_w}).to_csv(
    r"G:\OPT\q-Weibull竞争风险论文\data\estimates.csv", index=False, encoding="utf-8-sig")
print("\n结果已保存。")
