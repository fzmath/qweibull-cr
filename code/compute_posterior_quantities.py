# -*- coding: utf-8 -*-
"""后验派生量: 每模式 CIF, 系统可靠度 R_sys, 原因概率 (含后验不确定性)。"""
import numpy as np
from qweibull import sf, pdf
np.set_printoptions(precision=3, suppress=True)

ch = np.load(r"G:\OPT\q-Weibull竞争风险论文\data\post_chains.npy")
C, N, d = ch.shape
post = ch[:, 1000:, :].reshape(-1, d)
# 抽样子样本
rng = np.random.default_rng(3)
idx = rng.choice(len(post), size=400, replace=False)
S = post[idx]

T0 = 0.2
tg = np.linspace(T0, 3000.0, int((3000.0-T0)/0.4)+1)
dt = np.diff(tg)
M = len(S)
Rsys = np.zeros((M, len(tg)))
CIF = np.zeros((2, M, len(tg)))

for m, x in enumerate(S):
    q = [2-np.exp(x[0]), 2-np.exp(x[3])]
    eta = [np.exp(x[1]), np.exp(x[4])]
    beta = [np.exp(x[2]), np.exp(x[5])]
    lam = [eta[j]**(-beta[j]) for j in range(2)]
    R = [sf(q[j], lam[j], beta[j], tg) for j in range(2)]
    f = [pdf(q[j], lam[j], beta[j], tg) for j in range(2)]
    R = [np.nan_to_num(R[j], nan=0.0, posinf=0.0) for j in range(2)]
    f = [np.nan_to_num(f[j], nan=0.0, posinf=0.0) for j in range(2)]
    Rsys[m] = R[0]*R[1]
    for j in range(2):
        integrand = f[j]*R[1-j]
        # 首区间 [0,T0] 解析积分 (f~(2-q)beta*lam*t^(beta-1)), 其余梯形
        first = (2-q[j])*lam[j]*T0**beta[j]*R[1-j][0]
        seg = np.cumsum(0.5*(integrand[1:]+integrand[:-1])*dt)
        CIF[j, m, 0] = first
        CIF[j, m, 1:] = first + seg

def band(A):
    return A.mean(0), np.percentile(A, 2.5, axis=0), np.percentile(A, 97.5, axis=0)

rm, rl, rh = band(Rsys)
cif_stats = [band(CIF[j]) for j in range(2)]

# 原因概率 = CIF 在 T_end; 残余 = R_sys(T_end)
pcause = np.stack([CIF[j, :, -1] for j in range(2)], axis=1)
resid = Rsys[:, -1]
tot = pcause[:,0]+pcause[:,1]+resid       # 逐draw 恒等式校验
print("T_end=3000 时:")
print(f"  逐draw 恒等式 mean={tot.mean():.4f} max|dev|={np.abs(tot-1).max():.4f}")
print(f"  P(原因E) mean={pcause[:,0].mean():.3f} median={np.median(pcause[:,0]):.3f} "
      f"[{np.percentile(pcause[:,0],2.5):.3f},{np.percentile(pcause[:,0],97.5):.3f}]")
print(f"  P(原因D) mean={pcause[:,1].mean():.3f} median={np.median(pcause[:,1]):.3f} "
      f"[{np.percentile(pcause[:,1],2.5):.3f},{np.percentile(pcause[:,1],97.5):.3f}]")
print(f"  残余 R_sys(3000) = {resid.mean():.5f}")

# 观测窗口(446)处的 CIF/R
i446 = np.argmin(np.abs(tg-446))
t446 = CIF[0,:,i446]+CIF[1,:,i446]+Rsys[:,i446]
print(f"\n在 t=446: 恒等式mean={t446.mean():.4f} R_sys={rm[i446]:.3f} "
      f"CIF_E={cif_stats[0][0][i446]:.3f} CIF_D={cif_stats[1][0][i446]:.3f}")

np.savez(r"G:\OPT\q-Weibull竞争风险论文\data\posterior_quantities.npz",
         tg=tg, rsys_mean=rm, rsys_lo=rl, rsys_hi=rh,
         cif_e_mean=cif_stats[0][0], cif_e_lo=cif_stats[0][1], cif_e_hi=cif_stats[0][2],
         cif_d_mean=cif_stats[1][0], cif_d_lo=cif_stats[1][1], cif_d_hi=cif_stats[1][2],
         pcause=pcause, resid=resid)
print("已保存 posterior_quantities.npz")
