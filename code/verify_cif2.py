# -*- coding: utf-8 -*-
"""用 scipy.quad 高精度基准锚定 CIF, 并验证'首区间解析+梯形'网格法。"""
import numpy as np
from scipy.integrate import quad
from qweibull import sf, pdf

ch = np.load(r"G:\OPT\q-Weibull竞争风险论文\data\post_chains.npy")
post = ch[:, 1000:, :].reshape(-1, ch.shape[2])
rng = np.random.default_rng(3)
S = post[rng.choice(len(post), size=30, replace=False)]
TEND, T0 = 3000.0, 0.2

def params(x):
    q=[2-np.exp(x[0]),2-np.exp(x[3])]; eta=[np.exp(x[1]),np.exp(x[4])]
    b=[np.exp(x[2]),np.exp(x[5])]; lam=[eta[j]**(-b[j]) for j in range(2)]
    return q,lam,b

res_quad=[]; res_grid=[]
for x in S:
    q,lam,b = params(x)
    R=[lambda t,j=j: sf(q[j],lam[j],b[j],np.array([t]))[0] for j in range(2)]
    f=[lambda t,j=j: pdf(q[j],lam[j],b[j],np.array([t]))[0] for j in range(2)]
    # --- quad 基准 ---
    ce=quad(lambda t: f[0](t)*R[1](t),0,TEND,limit=200,epsabs=1e-7)[0]
    cd=quad(lambda t: f[1](t)*R[0](t),0,TEND,limit=200,epsabs=1e-7)[0]
    rT=R[0](TEND)*R[1](TEND)
    res_quad.append((ce,cd,rT))
    # --- 网格: 首区间[0,T0]解析, 其余梯形 dt=0.25 ---
    tg=np.linspace(T0,TEND,int((TEND-T0)/0.25)+1); dt=np.diff(tg)
    Rg=[np.nan_to_num(sf(q[j],lam[j],b[j],tg)) for j in range(2)]
    fg=[np.nan_to_num(pdf(q[j],lam[j],b[j],tg)) for j in range(2)]
    ge=[]; gd=[]
    for j in (0,1):
        ig=fg[j]*Rg[1-j]
        first=(2-q[j])*lam[j]*T0**b[j]*Rg[1-j][0]   # 解析 ∫0^T0
        body=np.sum(0.5*(ig[1:]+ig[:-1])*dt)
        (ge if j==0 else gd).append(first+body)
    res_grid.append((ge[0],gd[0],Rg[0][-1]*Rg[1][-1]))

res_quad=np.array(res_quad); res_grid=np.array(res_grid)
for nm,r in [("quad",res_quad),("grid",res_grid)]:
    print(f"{nm}: P_E={r[:,0].mean():.4f} P_D={r[:,1].mean():.4f} "
          f"R={r[:,2].mean():.5f} sum={(r.sum(1)).mean():.4f}")
d=res_grid-res_quad
print(f"grid-quad 差: E均值{d[:,0].mean():.5f} maxabs={np.abs(d[:,:2]).max():.5f}")
