# -*- coding: utf-8 -*-
"""中截断单数据集: q 剖面似然, 看窄峰/宽坪结构。"""
import sys
import numpy as np
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
from qweibull import Q_MIN, Q_MAX
from bayes_model import _loglik_and_grad_raw
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

rng=np.random.default_rng(2026)
T0=rnd_qw(1.45,76.7,2,200,rng); T1=rnd_qw(1,113,3,200,rng)
t=np.minimum(T0,T1); cause=(T1<T0).astype(int)
free0=np.array([np.log(76.7**-2),np.log(2),np.log(1),np.log(113.0**-3),np.log(3)])
print(" qfix   负LL   相对峰",flush=True)
rows=[]
for qfix in np.arange(0.80,1.72,0.06):
    def obj(free,qf=qfix):
        theta=np.array([np.log(2-qf),free[0],free[1],free[2],free[3],free[4]])
        v,g=cr_ng(theta,t,cause); return v,g[1:]
    r=minimize(obj,free0,jac=True,method="BFGS",options=dict(maxiter=400))
    rows.append((qfix,r.fun))
m=min(x[1] for x in rows)
for q,v in rows:
    print(f"{q:.2f}   {v:.3f}   {v-m:+.3f}",flush=True)
