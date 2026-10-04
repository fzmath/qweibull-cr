# -*- coding: utf-8 -*-
"""深度判别: 有界弱截断数据, 固定 eps/质量, 无 warmup 手动 NUTS, 看链是否移动。"""
import sys
import numpy as np
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
from qweibull import Q_MIN, Q_MAX
from bayes_model import _loglik_and_grad_raw
from cr_bayes_model import default_cr_prior, cr_logpost_grad
from nuts import numerical_hessian, mass_from_hessian, _one_transition
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
C,Sinv,Lc=mass_from_hessian(H0)
print("MLE xe q方向:", np.round(xe[0],3), " lp=", round(gradf(xe)[0],2))

for eps in [0.01, 0.03, 0.1]:
    np.random.seed(123)
    x=xe.copy(); lp,g=gradf(x); qs=[]; aa=0; nd=0
    for i in range(400):
        x,g,lp,dep,ai,nai,ndi=_one_transition(x,g,lp,eps,Sinv,Lc,gradf)
        qs.append(2-np.exp(x[0])); aa+=ai/nai; nd+=ndi
    qs=np.array(qs)
    print(f"eps={eps}: q mean={qs.mean():.3f} sd={qs.std():.3f} "
          f"范围=({qs.min():.3f},{qs.max():.3f}) alpha={aa/400:.3f} 发散={nd}")
