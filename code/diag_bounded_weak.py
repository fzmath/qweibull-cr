# -*- coding: utf-8 -*-
"""诊断: 有界-弱截断 q=0.8, 充分 NUTS, 看 q 后验 Rhat/ESS/发散/CI 是否覆盖真值。"""
import sys
import numpy as np
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
from qweibull import Q_MIN, Q_MAX
from bayes_model import _loglik_and_grad_raw
from cr_bayes_model import default_cr_prior, cr_logpost_grad
from nuts import sample_nuts, numerical_hessian
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
print("目标事件数=", (cause==0).sum(), " 竞争事件数=", (cause==1).sum())
priors=default_cr_prior(t,cause)
def gradf(x): return cr_logpost_grad(x,t,cause,priors)
x0=np.concatenate([[np.log(1),np.log(100.0**-2),np.log(2)],
                   [np.log(1),np.log(400.0**-2),np.log(2)]])
th=minimize(lambda z: cr_ng(z,t,cause),x0,jac=True,method="BFGS").x
xe=[]
for j in range(2):
    xq,xl,xb=th[3*j:3*j+3]; ll=np.exp(xl); bb=np.exp(xb)
    xe+=[xq,np.log(ll**(-1/bb)),xb]
xe=np.array(xe)
H0=-numerical_hessian(gradf,xe,1e-4)
inits=[xe+0.05*rng.standard_normal(6) for _ in range(4)]
d=sample_nuts(inits,gradf,n_draw=1500,warmup=500,target_accept=0.9,init_hessian=H0)
p=d["chains"][:,500:,:]
for c in range(4):
    qq=2-np.exp(p[c,:,0])
    print(f"链{c}: q mean={qq.mean():.3f} CI=({np.percentile(qq,2.5):.3f},"
          f"{np.percentile(qq,97.5):.3f})")
print("q Rhat=", round(float(d["rhat"][0]),3), " ESS=", int(d["ess"][0]),
      " 发散=", d["divergent"], " 接受率=", round(d["accept"],3),
      " 树深=", round(d["depth"],2))
print("真值 q=0.8")
