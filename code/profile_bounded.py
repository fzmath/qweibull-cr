# -*- coding: utf-8 -*-
"""有界弱截断数据: 目标模式 q 的剖面似然, 判断 q 是否可识别。"""
import sys
import numpy as np
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
from qweibull import Q_MIN, Q_MAX
from bayes_model import _loglik_and_grad_raw
from nuts import numerical_hessian
from scipy.optimize import minimize

def rnd_qw(q,eta,beta,n,rng):
    u=np.maximum(rng.random(n),1e-12); lam=eta**-beta
    if abs(q-1)<1e-9: return eta*(-np.log(u))**(1/beta)
    d=1-q; return ((1-u**(1/((2-q)/d)))/(d*lam))**(1/beta)
def cr_ng(theta,t,cause):
    neg=0; g=np.zeros(6)
    for j in range(2):
        xq,xl,xb=theta[3*j:3*j+3]; q=2-np.exp(xq); lam=np.exp(xl); beta=np.exp(xb)
        if q<Q_MIN or q>Q_MAX: return 1e10,np.zeros(6)
        ll,gq,gl,gb=_loglik_and_grad_raw(q,lam,beta,t,(cause==j).astype(int))
        if not np.isfinite(ll): return 1e10,np.zeros(6)
        neg+=ll; g[3*j]=gq*-np.exp(xq); g[3*j+1]=gl*lam; g[3*j+2]=gb*beta
    return -neg,-g

rng=np.random.default_rng(99)
T0=rnd_qw(0.8,100,2,200,rng); T1=rnd_qw(1,400,2,200,rng)
t=np.minimum(T0,T1); cause=(T1<T0).astype(int)

free0=np.array([np.log(100.0**-2),np.log(2.0), np.log(1),np.log(400.0**-2),np.log(2)])
print(" qfix    剖面负LL", flush=True)
best=(None,1e18)
for qfix in np.arange(0.60,1.15,0.10):
    def obj(free, qf=qfix):
        theta=np.array([np.log(2-qf),free[0],free[1],free[2],free[3],free[4]])
        v,g=cr_ng(theta,t,cause)
        return v, g[1:]
    r=minimize(obj,free0,jac=True,method="BFGS",options=dict(maxiter=300))
    if r.fun<best[1]: best=(qfix,r.fun)
    print(f"{qfix:.2f}    {r.fun:.3f}", flush=True)
print(f"剖面最优 q*={best[0]:.2f} 负LL={best[1]:.3f}  (真值 q=0.8)", flush=True)
