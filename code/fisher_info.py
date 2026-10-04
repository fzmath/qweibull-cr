# -*- coding: utf-8 -*-
"""审稿人 P1: 弱可识别性定量图谱。
对目标单模式 (cause-specific, 目标事件=事件, 其余=删失), 在真实参数处计算
观测 Fisher 信息矩阵 I(theta), theta=(q, log eta, log beta) 尺度。
报 det(I) 与条件数 kappa=lam_max/lam_min; 并与已有 sim_grid 的 MLE/Bayes RMSE 对照。
画 weak identifiability map -> figures/fig7_weak_id_map.{pdf,png}。
写 data/fisher_info.json。"""
import sys, os, json
import numpy as np
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
from qweibull import log_pdf, log_sf
from bayes_model import _loglik_and_grad_raw
from nuts import numerical_hessian
from sim_grid import median_scale, rnd_qw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ET, BT, BC = 100.0, 2.0, 3.0
n = 200
Q_GRID = [0.6, 0.8, 1.0, 1.25, 1.45, 1.7]
RHOS = [("strong", 0.6), ("moderate", 1.0), ("weak", 4.0)]
DATADIR = r"G:\OPT\q-Weibull竞争风险论文\data\sim_grid"
OUTJSON = r"G:\OPT\q-Weibull竞争风险论文\data\fisher_info.json"
FIGDIR = r"G:\OPT\q-Weibull竞争风险论文\figures"


def cs_gradf_factory(t, status_E):
    """构造 theta=(q, xe=log eta, xb=log beta) -> (loglik, grad) 的 cause-specific 子似然。
    lam = eta^(-beta) = exp(-beta*xe); beta=exp(xb)。"""
    def gradf(theta):
        q, xe, xb = theta
        beta = np.exp(xb)
        lam = np.exp(-beta * xe)
        ll, gq, gl, gb = _loglik_and_grad_raw(q, lam, beta, t, status_E)
        if not np.isfinite(ll):
            return -np.inf, np.zeros(3)
        g_xe = -beta * lam * gl
        g_xb = beta * (gb - gl * lam * xe)
        return ll, np.array([gq, g_xe, g_xb])
    return gradf


def fisher_cell(qtrue, rho):
    rng = np.random.default_rng(20260504)
    tmed_t = ET * median_scale(qtrue, BT)
    eta_c = rho * tmed_t / median_scale(1, BC)
    T0 = rnd_qw(qtrue, ET, BT, n, rng)
    T1 = rnd_qw(1.0, eta_c, BC, n, rng)
    t = np.minimum(T0, T1); cause = (T1 < T0).astype(int)
    status_E = (cause == 0).astype(int)
    theta_true = np.array([qtrue, np.log(ET), np.log(BT)])
    gradf = cs_gradf_factory(t, status_E)
    H = numerical_hessian(gradf, theta_true, 1e-4)   # Hessian of loglik
    I = -H
    I = 0.5 * (I + I.T)
    w = np.linalg.eigvalsh(I)
    w = np.sort(w)
    detI = float(np.prod(w))
    wmin = max(w[0], 1e-12); wmax = w[-1]
    kappa = float(wmax / wmin)
    frac_event = float(status_E.mean())
    return dict(detI=detI, kappa=kappa, eigmin=float(w[0]), eigmax=float(wmax),
                frac_event=frac_event)


def main():
    records = []
    for sname, rho in RHOS:
        for q in Q_GRID:
            f = fisher_cell(q, rho)
            # 读已有 sim_grid 的 RMSE
            fp = os.path.join(DATADIR, f"sim_q{q}_{sname}.json")
            sg = json.load(open(fp, encoding="utf-8"))
            rec = dict(rho_name=sname, rho=rho, q=q,
                       detI=f["detI"], kappa=f["kappa"],
                       eigmin=f["eigmin"], eigmax=f["eigmax"],
                       frac_event=f["frac_event"],
                       mle_rmse=sg["mle"]["rmse"], bayes_rmse=sg["bayes"]["rmse"],
                       mle_bias=sg["mle"]["bias"], bayes_bias=sg["bayes"]["bias"])
            records.append(rec)
            print(f"rho={rho} q={q}: kappa={f['kappa']:.2e} detI={f['detI']:.2e} "
                  f"fracE={f['frac_event']:.2f} mleRMSE={rec['mle_rmse']:.3f} "
                  f"bayesRMSE={rec['bayes_rmse']:.3f}", flush=True)
    json.dump(dict(n=n, ET=ET, BT=BT, BC=BC, records=records),
              open(OUTJSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    make_plot(records)
    print("WROTE", OUTJSON)


def make_plot(records):
    qs = sorted(set(r["q"] for r in records))
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    markers = ["o", "s", "^", "D", "v", "P"]
    colors = {q: palette[i % len(palette)] for i, q in enumerate(qs)}
    markmap = {q: markers[i % len(markers)] for i, q in enumerate(qs)}
    rho_x = {0.6: 0.6, 1.0: 1.0, 4.0: 4.0}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=True)
    for q in qs:
        rows = sorted([r for r in records if r["q"] == q], key=lambda r: r["rho"])
        mk = markmap[q]; col = colors[q]
        # 左: x=rho(log)
        xl = [np.log10(rho_x[r["rho"]]) for r in rows]
        axL.plot(xl, [r["mle_rmse"] for r in rows], "-", color=col,
                 marker=mk, mfc="white", mec=col, mew=1.4, lw=1.3)
        axL.plot(xl, [r["bayes_rmse"] for r in rows], "--", color=col,
                 marker=mk, mfc=col, mec=col, mew=1.0, lw=1.3)
        for r in rows:
            axL.annotate(f"{r['kappa']:.0f}",
                         (np.log10(rho_x[r["rho"]]), r["bayes_rmse"]),
                         textcoords="offset points", xytext=(3, 5),
                         fontsize=5.5, color=col)
        # 右: x=log10(kappa)
        xr = [np.log10(r["kappa"]) for r in rows]
        axR.plot(xr, [r["mle_rmse"] for r in rows], "-", color=col,
                 marker=mk, mfc="white", mec=col, mew=1.4, lw=1.3)
        axR.plot(xr, [r["bayes_rmse"] for r in rows], "--", color=col,
                 marker=mk, mfc=col, mec=col, mew=1.0, lw=1.3)
    axL.set_xticks([np.log10(0.6), np.log10(1.0), np.log10(4.0)])
    axL.set_xticklabels(["strong\nrho=0.6", "moderate\nrho=1.0", "weak\nrho=4.0"])
    axL.set_xlabel("competing-risk cutoff strength  (log rho)")
    axL.set_ylabel("RMSE of q_E estimate")
    axL.set_title("(a) RMSE by cutoff strength")
    axR.set_xlabel(r"$\log_{10}\,\kappa(I)$")
    axR.set_title("(b) RMSE vs Fisher conditioning")
    for ax in (axL, axR):
        ax.grid(alpha=0.3)
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color="k", lw=1.1, marker="o", mfc="white", mec="k", label="MLE q_W CR"),
        Line2D([0], [0], color="k", lw=1.1, ls="--", marker="o", mfc="k", mec="k",
               label="Bayes q_W CR (median)"),
    ]
    axL.legend(handles=legend_elems, loc="upper left", fontsize=8, framealpha=0.9)
    q_elems = [Line2D([0], [0], color=colors[q], lw=1.8, marker=markmap[q],
                      mfc=colors[q], mec=colors[q], label=f"q={q}") for q in qs]
    axR.legend(handles=q_elems, loc="upper right", fontsize=7, title="true q_E",
               title_fontsize=8, framealpha=0.9)
    fig.suptitle("Weak identifiability map (open=MLE, filled=Bayes; numbers on (a) = kappa(I))",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"fig7_weak_id_map.{ext}"), dpi=300)
    print("WROTE fig7")


if __name__ == "__main__":
    main()
