# -*- coding: utf-8 -*-
"""P3 诊断: 参数化稳健的 q 可识别性指标 (替代尺度依赖的 det(I))。

对每个截断比 rho in {strong=0.6, moderate=1.0, weak=4.0}、
代表性 qtrue in {0.6, 1.0, 1.45}, 在 ET=100, BT=2, BC=3, n=200 的
目标 cause-specific 子模型 (cause==E=事件, 其余=删失) 真实参数处:

1) Schur complement: 3x3 Fisher 信息 I(q, log eta, log beta),
   把 (log eta, log beta) 当 nuisance, 报
       I_{q.psi} = I_qq - I_qpsi I_psi^-1 I_psiq,
       渐近 SE_q = 1/sqrt(I_{q.psi})   (尺度稳健, 随 rho 的变化反映 q 的真实可识别性)。
2) Profile-likelihood width: 沿 q profile (每 q 优化 eta, beta),
   W_q = length{ q : 2[ell(qhat)-ell(q)] <= chi2_{1,0.95}=3.841 }。

数据生成与 fisher_info.py 完全一致 (rng=default_rng(20260504)),
复用其 cs_gradf_factory。结果写 data/identifiability_metrics.json。
"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
from nuts import numerical_hessian
from fisher_info import cs_gradf_factory
from sim_grid import median_scale, rnd_qw
from scipy.optimize import minimize

ET, BT, BC, n = 100.0, 2.0, 3.0, 200
RHOS = [("strong", 0.6), ("moderate", 1.0), ("weak", 4.0)]
QS = [0.6, 1.0, 1.45]
CHI2_95_1 = 3.8414588
OUTJSON = r"G:\OPT\q-Weibull竞争风险论文\data\identifiability_metrics.json"


def profile_ll_factory(t, status_E):
    """返回 (q) -> max_{xe,xb} loglik 的 profile 函数 (cause-specific 子模型)。
    用有界 L-BFGS-B (带梯度), 防止优化器在 q->2 时把 beta 跑飞到 _soft_z 截断区
    产生 ll 数值伪影。bounds: eta in [10,2000], beta in [0.4,8]。"""
    gradf = cs_gradf_factory(t, status_E)
    bnds = [(np.log(10.0), np.log(2000.0)),
            (np.log(0.4), np.log(8.0))]

    def neg_jac(z, q):
        ll, g = gradf(np.array([q, z[0], z[1]]))
        if not np.isfinite(ll):
            return 1e12, np.zeros(2)
        return -ll, -g[1:]

    def best_ll(q):
        best = -np.inf
        for x0 in [(np.log(ET), np.log(BT)),
                   (np.log(0.5 * ET), np.log(1.0)),
                   (np.log(2.0 * ET), np.log(3.0))]:
            r = minimize(neg_jac, x0, args=(q,), jac=True, method="L-BFGS-B",
                         bounds=bnds,
                         options=dict(maxiter=500, ftol=1e-9))
            if -r.fun > best:
                best = -r.fun
        return best
    return best_ll


def cell(qtrue, rho):
    rng = np.random.default_rng(20260504)     # 与 fisher_info.py 同种子
    tmed_t = ET * median_scale(qtrue, BT)
    eta_c = rho * tmed_t / median_scale(1, BC)
    T0 = rnd_qw(qtrue, ET, BT, n, rng)
    T1 = rnd_qw(1.0, eta_c, BC, n, rng)
    t = np.minimum(T0, T1)
    cause = (T1 < T0).astype(int)
    status_E = (cause == 0).astype(int)
    frac_event = float(status_E.mean())

    gradf = cs_gradf_factory(t, status_E)
    theta_true = np.array([qtrue, np.log(ET), np.log(BT)])

    # ---- 1) Schur complement of Fisher info ----
    H = numerical_hessian(gradf, theta_true, 1e-4)
    I = -H
    I = 0.5 * (I + I.T)
    Iqq = I[0, 0]
    Iqpsi = I[0, 1:]
    Ipsi = I[1:, 1:]
    schur = Iqq - Iqpsi @ np.linalg.solve(Ipsi, Iqpsi)
    schur = float(schur)
    schur_se = float(1.0 / np.sqrt(max(schur, 1e-12)))

    # ---- 2) Profile-likelihood width W_q ----
    best_ll = profile_ll_factory(t, status_E)
    qlo = max(0.25, qtrue - 1.3)
    qhi = min(1.85, qtrue + 1.3)   # q>1.85 进入 _soft_z 数值不稳区, 不纳入
    grid = np.round(np.arange(qlo, qhi + 1e-9, 0.025), 4)
    lls = np.array([best_ll(q) for q in grid])
    imax = int(np.argmax(lls))
    llmax = lls[imax]
    qhat = float(grid[imax])
    # 联合精修 MLE (qhat 附近, 3 参数, 有界)
    def neg3(z):
        ll, g = gradf(z)
        return (-ll if np.isfinite(ll) else 1e12), -g
    r3 = minimize(neg3, np.array([qhat, np.log(ET), np.log(BT)]),
                  jac=True, method="L-BFGS-B",
                  bounds=[(0.25, 1.85), (np.log(10.), np.log(2000.)),
                          (np.log(0.4), np.log(8.))],
                  options=dict(maxiter=1000, ftol=1e-10))
    if -r3.fun > llmax:
        llmax = -r3.fun
        qhat = float(r3.x[0])

    dev = 2.0 * (llmax - lls)
    inside = dev <= CHI2_95_1
    # 找包含 qhat 的连通区间
    li = imax
    while li > 0 and inside[li - 1]:
        li -= 1
    ri = imax
    while ri < len(grid) - 1 and inside[ri + 1]:
        ri += 1
    one_sided = False
    # 端点线性插值细化
    def interp(lo_i, hi_i, target):
        x1, y1 = grid[lo_i], dev[lo_i]
        x2, y2 = grid[hi_i], dev[hi_i]
        if abs(y2 - y1) < 1e-12:
            return x2
        return x1 + (target - y1) * (x2 - x1) / (y2 - y1)
    if li > 0:
        q_left = interp(li - 1, li, CHI2_95_1)
    else:
        q_left = float(grid[0]); one_sided = True
    if ri < len(grid) - 1:
        q_right = interp(ri, ri + 1, CHI2_95_1)
    else:
        q_right = float(grid[-1]); one_sided = True
    Wq = float(q_right - q_left)

    return dict(qtrue=qtrue, rho=rho, eta_c=float(eta_c),
                frac_event=frac_event,
                Schur_I_q=schur, Schur_se_q=schur_se,
                qhat=qhat, profile_width_Wq=Wq,
                q_ci_left=float(q_left), q_ci_right=float(q_right),
                one_sided_ci=bool(one_sided))


def main():
    records = []
    for sname, rho in RHOS:
        for q in QS:
            t0 = time.time()
            rec = cell(q, rho)
            rec["rho_name"] = sname
            records.append(rec)
            print(f"rho={rho:<4}({sname:<8}) qtrue={q}: "
                  f"I_q.psi={rec['Schur_I_q']:.3f}  se={rec['Schur_se_q']:.4f}  "
                  f"Wq={rec['profile_width_Wq']:.3f}  "
                  f"fracE={rec['frac_event']:.3f}  [{time.time()-t0:.1f}s]",
                  flush=True)
    out = dict(n=n, ET=ET, BT=BT, BC=BC,
               note="cause-specific target submodel; I(q,log eta,log beta); "
                    "Schur complements out (log eta,log beta); "
                    "W_q = 95% profile-LR interval width, chi2_1=3.841",
               records=records)
    json.dump(out, open(OUTJSON, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("WROTE", OUTJSON)


if __name__ == "__main__":
    main()
