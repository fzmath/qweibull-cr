# -*- coding: utf-8 -*-
"""诊断: 逐draw验证 CIF_E+CIF_D+R_sys=1, 对比粗/密网格, 定位积分误差。"""
import numpy as np
from qweibull import sf, pdf

ch = np.load(r"G:\OPT\q-Weibull竞争风险论文\data\post_chains.npy")
C, N, d = ch.shape
post = ch[:, 1000:, :].reshape(-1, d)
rng = np.random.default_rng(3)
S = post[rng.choice(len(post), size=400, replace=False)]
M = len(S)

def run(tg):
    dt = np.diff(tg)
    Rsys = np.zeros(M); CIF = np.zeros((2, M))
    for m, x in enumerate(S):
        q = [2-np.exp(x[0]), 2-np.exp(x[3])]
        eta = [np.exp(x[1]), np.exp(x[4])]
        beta = [np.exp(x[2]), np.exp(x[5])]
        lam = [eta[j]**(-beta[j]) for j in range(2)]
        R = [np.nan_to_num(sf(q[j],lam[j],beta[j],tg)) for j in range(2)]
        f = [np.nan_to_num(pdf(q[j],lam[j],beta[j],tg)) for j in range(2)]
        Rsys[m] = R[0][-1]*R[1][-1]
        for j in range(2):
            ig = f[j]*R[1-j]
            CIF[j,m] = np.sum(0.5*(ig[1:]+ig[:-1])*dt)
    return Rsys, CIF

for label, tg in [("coarse 600", np.linspace(0.05,3000,600)),
                  ("dense 6000", np.linspace(0.0,3000,6000))]:
    Rsys, CIF = run(tg)
    tot = CIF[0]+CIF[1]+Rsys          # 逐draw 恒等式
    print(f"--- {label} ---")
    print(f" mean P_E={CIF[0].mean():.4f}  P_D={CIF[1].mean():.4f}  "
          f"Rsys={Rsys.mean():.4f}")
    print(f" mean P_E+P_D+Rsys = {tot.mean():.4f}   "
          f"draw-wise dev: max={np.abs(tot-1).max():.4f} "
          f"p95={np.percentile(np.abs(tot-1),95):.4f}")
    i446 = np.argmin(np.abs(tg-446))
    # 446 处
    R446 = np.zeros(M); C446 = np.zeros((2,M))
    dt = np.diff(tg[:i446+1])
    for m,x in enumerate(S):
        q=[2-np.exp(x[0]),2-np.exp(x[3])]; eta=[np.exp(x[1]),np.exp(x[4])]
        beta=[np.exp(x[2]),np.exp(x[5])]; lam=[eta[j]**(-beta[j]) for j in range(2)]
        R=[np.nan_to_num(sf(q[j],lam[j],beta[j],tg[:i446+1])) for j in range(2)]
        f=[np.nan_to_num(pdf(q[j],lam[j],beta[j],tg[:i446+1])) for j in range(2)]
        R446[m]=R[0][-1]*R[1][-1]
        for j in range(2):
            ig=f[j]*R[1-j]; C446[j,m]=np.sum(0.5*(ig[1:]+ig[:-1])*dt)
    print(f" @446 mean: R={R446.mean():.4f} CIF_E={C446[0].mean():.4f} "
          f"CIF_D={C446[1].mean():.4f} sum={ (R446+C446[0]+C446[1]).mean():.4f}")
