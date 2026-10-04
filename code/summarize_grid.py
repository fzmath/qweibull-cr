# -*- coding: utf-8 -*-
"""汇总 sim_grid 18 组合 R=80 结果, 打印总表 (供论文 Table/fig4)。"""
import os, json
import numpy as np

OUTDIR = r"G:\OPT\q-Weibull竞争风险论文\data\sim_grid"
Q_GRID = [0.6, 0.8, 1.0, 1.25, 1.45, 1.7]
SCEN = [("strong", "Strong"), ("moderate", "Moderate"), ("weak", "Weak")]

tab = {}
for q in Q_GRID:
    for s, _ in SCEN:
        fp = os.path.join(OUTDIR, f"sim_q{q}_{s}.json")
        d = json.load(open(fp, encoding="utf-8"))
        tab[(q, s)] = d

print("q     scen      MLE bias  rmse  cov  | Bayes bias  rmse  cov  width  ess  div")
for q in Q_GRID:
    for s, lab in SCEN:
        d = tab[(q, s)]; m, b = d["mle"], d["bayes"]
        print(f"{q:<5} {lab:<9} {m['bias']:6.2f} {m['rmse']:5.2f} "
              f"{m['coverage']:4.2f} | {b['bias']:6.2f} {b['rmse']:5.2f} "
              f"{b['coverage']:4.2f} {b['width']:5.2f} {b['ess']:4.0f} "
              f"{b['divergences']:4.2f}")

# 关键对比: strong 下 Bayes vs MLE RMSE; weak 下
print("\n--- RMSE 比值 Bayes/MLE ---")
print("q      strong  moderate  weak")
for q in Q_GRID:
    r = [tab[(q, s)]["bayes"]["rmse"]/tab[(q, s)]["mle"]["rmse"] for s, _ in SCEN]
    print(f"{q:<5} " + "  ".join(f"{x:5.2f}" for x in r))
