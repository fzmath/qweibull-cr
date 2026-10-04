# -*- coding: utf-8 -*-
"""仿真(主线): 重尾 q=1.45 目标模式, 竞争 Weibull(beta=3) 截断强/中/弱。
MLE vs Bayes: q 的 偏差/RMSE/95覆盖率。NUTS 在重尾(无硬墙)下高效。"""
import sys, time, json
import numpy as np
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
from qweibull import Q_MIN, Q_MAX
from bayes_model import _loglik_and_grad_raw
from cr_bayes_model import default_cr_prior, cr_logpost_grad
from nuts import sample_nuts, numerical_hessian
from scipy.optimize import minimize

QT, ET, BT = 1.45, 76.7, 2.0          # 目标: 重尾, lam=eta^-beta≈1.7e-4
BC = 3.0                              # 竞争 Weibull 形状
scenarios = [("强截断(竞争早)", 68.0), ("中截断", 113.0), ("弱截断(竞争晚)", 452.0)]

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

n=200
R=int(sys.argv[1]) if len(sys.argv)>1 else 5
WU,PD=200,300
results={}
for name,eta2 in scenarios:
    rng=np.random.default_rng(2026)
    mq,bq,mc,bc,sec=[],[],[],[],0
    for r in range(R):
        T0=rnd_qw(QT,ET,BT,n,rng); T1=rnd_qw(1,eta2,BC,n,rng)
        t=np.minimum(T0,T1); cause=(T1<T0).astype(int)
        x0=np.concatenate([[np.log(2-QT),np.log(ET**-BT),np.log(BT)],
                           [np.log(1),np.log(eta2**-BC),np.log(BC)]])
        th=minimize(lambda z: cr_ng(z,t,cause),x0,jac=True,method="BFGS",
                    options=dict(maxiter=500)).x
        qm=2-np.exp(th[0]); mq.append(qm)
        H=numerical_hessian(lambda z: cr_ng(z,t,cause), th, 1e-3)
        try:
            Ci=np.linalg.inv(H+1e-6*np.eye(6)); se=np.exp(th[0])*np.sqrt(max(Ci[0,0],0))
            se=min(se,5); mc.append(qm-1.96*se<=QT<=qm+1.96*se)
        except Exception: mc.append(False)
        priors=default_cr_prior(t,cause)
        def gradf(x): return cr_logpost_grad(x,t,cause,priors)
        xe=[]
        for j in range(2):
            xq,xl,xb=th[3*j:3*j+3]; ll=np.exp(xl); bb=np.exp(xb)
            xe+=[xq,np.log(ll**(-1/bb)),xb]
        xe=np.array(xe)
        dg=sample_nuts([xe],gradf,n_draw=WU+PD,warmup=WU,target_accept=0.85,
                       init_hessian=-numerical_hessian(gradf,xe,1e-4))
        qp=2-np.exp(dg["chains"][0,WU:,0]); bq.append(np.median(qp))
        lo,hi=np.percentile(qp,[2.5,97.5]); bc.append(lo<=QT<=hi); sec+=dg["elapsed"]
    mq,bq=np.array(mq),np.array(bq)
    s=dict(qtrue=QT,
        mle_bias=float((mq-QT).mean()),mle_rmse=float(np.sqrt(((mq-QT)**2).mean())),
        mle_cover=float(np.mean(mc)),
        bay_bias=float((bq-QT).mean()),bay_rmse=float(np.sqrt(((bq-QT)**2).mean())),
        bay_cover=float(np.mean(bc)),sec=float(sec/R))
    results[name]=s
    print(f"{name}: MLE bias={s['mle_bias']:+.3f} RMSE={s['mle_rmse']:.3f} "
          f"覆盖={s['mle_cover']:.2f} | Bayes bias={s['bay_bias']:+.3f} "
          f"RMSE={s['bay_rmse']:.3f} 覆盖={s['bay_cover']:.2f} ({s['sec']:.1f}s/rep)",
          flush=True)

if R>=20:
    json.dump(results,open(r"G:\OPT\q-Weibull竞争风险论文\data\sim_results_heavy.json","w",
              encoding="utf-8"),ensure_ascii=False,indent=2)
    print("已保存 sim_results_heavy.json")
