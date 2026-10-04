"""Compute complete posterior summaries for Table 3 from post_chains.npy."""
import numpy as np

path = r"G:\OPT\q-Weibull竞争风险论文\data\post_chains.npy"
ch = np.load(path)              # (C, total_draws, 6) in x coords
WARM = 1000
post = ch[:, WARM:, :]          # post-warmup
C, N, _ = post.shape
print("chains, post draws/chain:", C, N)

flat = post.reshape(-1, 6)
# x order: [xq_E, xe_E, xb_E, xq_D, xe_D, xb_D]
qE = 2 - np.exp(flat[:, 0]); etE = np.exp(flat[:, 1]); beE = np.exp(flat[:, 2])
qD = 2 - np.exp(flat[:, 3]); etD = np.exp(flat[:, 4]); beD = np.exp(flat[:, 5])

def summ(c, name, unit=False):
    lo, hi = np.percentile(c, [2.5, 97.5])
    print(f"{name:<6} median={np.median(c):8.3f}  mean={c.mean():8.3f}  "
          f"95%CI=[{lo:7.3f},{hi:7.3f}]")

summ(qE, "q_E"); summ(etE, "eta_E"); summ(beE, "beta_E")
summ(qD, "q_D"); summ(etD, "eta_D"); summ(beD, "beta_D")
print(f"P(q_E>1)={np.mean(qE>1):.3f}   P(q_D>1)={np.mean(qD>1):.3f}")
