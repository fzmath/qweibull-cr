# -*- coding: utf-8 -*-
"""效率对比: NUTS vs 预条件随机游走 MH (ESS/秒, 接受率, 树深/发散)。
在固定数据集上, 代表性场景: 重尾强/弱截断, q=1 对照。"""
import sys, time, json
import numpy as np
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
from qweibull import Q_MIN, Q_MAX
from bayes_model import _loglik_and_grad_raw
from cr_bayes_model import default_cr_prior, cr_logpost_grad
from nuts import (sample_nuts, numerical_hessian, mass_from_hessian,
                  _rhat, _ess)
from scipy.optimize import minimize

def rnd_qw(q, eta, beta, n, rng):
    u = np.maximum(rng.random(n), 1e-12); lam = eta**(-beta)
    if abs(q-1) < 1e-9:
        return eta*(-np.log(u))**(1/beta)
    d=1-q; p=(2-q)/d
    return ((1-u**(1/p))/(d*lam))**(1/beta)

def cr_mle_neggrad(theta, t, cause):
    neg=0.0; g=np.zeros(6)
    for j in range(2):
        xq,xl,xb=theta[3*j:3*j+3]
        q=2-np.exp(xq); lam=np.exp(xl); beta=np.exp(xb)
        if q<Q_MIN or q>Q_MAX: return 1e10, np.zeros(6)
        ll,gq,gl,gb=_loglik_and_grad_raw(q,lam,beta,t,(cause==j).astype(int))
        if not np.isfinite(ll): return 1e10, np.zeros(6)
        neg+=ll; g[3*j]=gq*-np.exp(xq); g[3*j+1]=gl*lam; g[3*j+2]=gb*beta
    return -neg,-g

def sample_mh(x0, gradf, n, warmup, Lprop, rng, target=0.234):
    x=x0.copy(); lp,_=gradf(x); out=[]; acc=0; sc=2.4/np.sqrt(len(x0))
    for i in range(n):
        xn=x+sc*(Lprop@rng.standard_normal(len(x0)))
        lpn,_=gradf(xn)
        if np.log(rng.random()) < lpn-lp:
            x,lp=xn,lpn; acc+=1
        if i < warmup and i % 40 == 39:
            sc *= np.exp((acc/40-target)*1.5)
            sc = np.clip(sc, 1e-3, 10.0); acc=0
        if i >= warmup:
            out.append(x)
    return np.array(out), acc/max(n-warmup,1), sc

scenarios = [("Heavy, strong trunc.",1.45,68.0),
             ("Heavy, weak trunc.",1.45,452.0),
             ("Standard Weibull",1.0,68.0)]
n=200
out_all={}
for name,qtrue,eta2 in scenarios:
    rng=np.random.default_rng(11)
    T0=rnd_qw(qtrue,76.7,2,n,rng); T1=rnd_qw(1,eta2,3,n,rng)
    t=np.minimum(T0,T1); cause=(T1<T0).astype(int)
    priors=default_cr_prior(t,cause)
    def gradf(x): return cr_logpost_grad(x,t,cause,priors)
    x0lam=np.concatenate([[np.log(1),np.log(76.7**-2),np.log(2)],
                          [np.log(1),np.log(eta2**-3),np.log(3)]])
    th=minimize(lambda z: cr_mle_neggrad(z,t,cause),x0lam,jac=True,
                method="BFGS").x
    xe=[]
    for j in range(2):
        xq,xl,xb=th[3*j:3*j+3]; ll=np.exp(xl); bb=np.exp(xb)
        xe+=[xq,np.log(ll**(-1/bb)),xb]
    xe=np.array(xe)
    H0=-numerical_hessian(gradf,xe,1e-4)
    C,_,_=mass_from_hessian(H0)
    Lprop=np.linalg.cholesky(C+1e-8*np.eye(6))  # 提议协方差=C(后验协方差)
    # NUTS
    dN=sample_nuts([xe,xe+0.05*rng.standard_normal(6)],gradf,n_draw=1000,
                   warmup=500,target_accept=0.85,init_hessian=H0)
    pN=dN["chains"][:,500:,:]
    # MH (预条件, 协方差=后验协方差, 给 MH 以公平优势)
    t0=time.time(); mh,accmh,sc=sample_mh(xe,gradf,4000,1000,Lprop,rng); tm=time.time()-t0
    pM=mh[None]
    essN=dN["ess"]; essM=_ess(pM)
    row=dict(
        nuts_ess=float(essN.sum()), nuts_sec=float(dN["elapsed"]),
        nuts_ess_sec=float(essN.sum()/dN["elapsed"]),
        nuts_accept=float(dN["accept"]), nuts_div=int(dN["divergent"]),
        mh_ess=float(essM.sum()), mh_sec=float(tm),
        mh_ess_sec=float(essM.sum()/tm), mh_accept=float(accmh))
    row["speedup"]=row["nuts_ess_sec"]/max(row["mh_ess_sec"],1e-9)
    out_all[name]=row
    print(f"{name}: NUTS ESS/s={row['nuts_ess_sec']:.1f} acc={row['nuts_accept']:.2f} "
          f"div={row['nuts_div']} | MH ESS/s={row['mh_ess_sec']:.1f} acc={row['mh_accept']:.2f} "
          f"| 加速x{row['speedup']:.1f}")

with open(r"G:\OPT\q-Weibull竞争风险论文\data\efficiency_results.json","w",
          encoding="utf-8") as f:
    json.dump(out_all,f,ensure_ascii=False,indent=2)
print("已保存 efficiency_results.json")
