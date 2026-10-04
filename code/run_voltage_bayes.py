# -*- coding: utf-8 -*-
"""voltage 数据 q-Weibull 竞争风险 Bayes NUTS 后验分析。"""
import numpy as np
import pandas as pd
from nuts import sample_nuts, numerical_hessian
from cr_bayes_model import default_cr_prior, cr_logpost_grad
np.set_printoptions(precision=3, suppress=True)

df = pd.read_csv(r"G:\OPT\q-Weibull竞争风险论文\data\voltage.csv")
t = df["hours"].to_numpy(float)
fm = df["failure_mode"].to_numpy(str)
cause = np.full(len(df), -1)
cause[fm == "E"] = 0
cause[fm == "D"] = 1
print("n=", len(df), " E=", (cause == 0).sum(),
      " D=", (cause == 1).sum(), " 删失=", (cause == -1).sum())

priors = default_cr_prior(t, cause)

def gradf(x):
    return cr_logpost_grad(x, t, cause, priors)

# MLE 初始点 (estimates)
mle = {
    0: dict(q=1.2414715, lam=0.01414739, beta=0.6550399),
    1: dict(q=0.9345138, lam=1.5741296e-14, beta=5.4219912),
}
def to_x(m):
    eta = m["lam"] ** (-1.0 / m["beta"])
    return np.array([np.log(2 - m["q"]), np.log(eta), np.log(m["beta"])])

x_mle = np.concatenate([to_x(mle[0]), to_x(mle[1])])
rng = np.random.RandomState(7)
inits = [x_mle, x_mle + 0.3 * rng.standard_normal(6)]

H0 = -numerical_hessian(gradf, x_mle, epsilon=1e-4)
print("初始 负Hessian 特征值:", np.linalg.eigvalsh(H0))

diag = sample_nuts(inits, gradf, n_draw=2000, warmup=1000,
                   target_accept=0.9, init_hessian=H0)

post = diag["chains"][:, 1000:, :]      # C, N, 6
def to_native(x):
    out = []
    for j in range(2):
        xq, xe, xb = x[..., 3*j], x[..., 3*j+1], x[..., 3*j+2]
        q = 2 - np.exp(xq); beta = np.exp(xb); eta = np.exp(xe)
        lam = eta ** (-beta)
        out += [q, beta, eta, lam]
    return np.stack(out, axis=-1)

nat = to_native(post)
names = ["q_E", "beta_E", "eta_E", "lam_E",
         "q_D", "beta_D", "eta_D", "lam_D"]
print("\n参数         后验均值   中位数     2.5%      97.5%")
flat = nat.reshape(-1, 8)
for k, nm in enumerate(names):
    c = flat[:, k]
    lo, hi = np.percentile(c, [2.5, 97.5])
    print(f"{nm:<8} {c.mean():10.4f} {np.median(c):10.4f} {lo:10.4f} {hi:10.4f}")

# q 后验含 1 的概率 (P(q>1))
for j, lab in [(0, "E"), (1, "D")]:
    qv = flat[:, j*4]
    print(f"P(q_{lab}>1) = {(qv > 1).mean():.3f}, P(|q_{lab}-1|<0.1) = "
          f"{(np.abs(qv-1)<0.1).mean():.3f}")

print("\nRhat =", diag["rhat"])
print("ESS  =", diag["ess"].astype(int))
print("树深 =", round(diag["depth"], 2), " 接受率 =", round(diag["accept"], 3),
      " post发散 =", diag["divergent"], " warmup发散 =", diag["divergent_warmup"])
print("耗时 =", round(diag["elapsed"], 1), "s, ESS/秒 =", round(diag["ess_per_sec"], 1))

np.save(r"G:\OPT\q-Weibull竞争风险论文\data\post_chains.npy",
        diag["chains"])
print("\n已保存后验链 post_chains.npy")
