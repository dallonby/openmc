"""Statistical comparison of two statepoints (CPU vs GPU differential)."""
import sys
import numpy as np
import openmc

sp_a = openmc.StatePoint(sys.argv[1])
sp_b = openmc.StatePoint(sys.argv[2])

print(f"k: A={sp_a.keff:.5uS}  B={sp_b.keff:.5uS}")

for tid, ta in sp_a.tallies.items():
    tb = sp_b.tallies[tid]
    ma, sa = ta.mean.ravel(), ta.std_dev.ravel()
    mb, sb = tb.mean.ravel(), tb.std_dev.ravel()
    sig = np.sqrt(sa**2 + sb**2)
    ok = sig > 0
    z = np.zeros_like(ma)
    z[ok] = (mb[ok] - ma[ok]) / sig[ok]
    rel = np.zeros_like(ma)
    nz = ma != 0
    rel[nz] = (mb[nz] - ma[nz]) / ma[nz]
    name = ta.name or f"tally {tid}"
    print(f"\n=== {name}: bins={ma.size} max|z|={np.abs(z).max():.2f} "
          f"mean z^2={np.mean(z[ok]**2):.2f}")
    # worst offenders
    order = np.argsort(-np.abs(z))[:8]
    scores = ta.scores
    n_s = len(scores)
    for idx in order:
        if abs(z[idx]) < 2:
            break
        b, s = divmod(idx, n_s)
        print(f"  bin {b:4d} {scores[s]:<12s} A={ma[idx]:.5e} "
              f"B={mb[idx]:.5e} rel={rel[idx]:+.4f} z={z[idx]:+.2f}")
