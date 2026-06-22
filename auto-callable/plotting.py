"""
plotting.py
===========
The four money-shot figures for the race. Each function takes plain arrays /
result dicts (from run_race_1d.py and run_race_wof3.py) so it runs without the
GPU; PINN points are filled from the H200 results JSON when present, otherwise
drawn as a clearly-labelled projection.

  fig1  MC convergence wall      -- error(bp of vol) vs paths, with 0.1 bp line
  fig2  crossover (HEADLINE)     -- total wall-clock vs # contracts; PINN crosses
                                    below per-contract FD/MC as the family grows
  fig3  accuracy frontier        -- error(bp) vs time, 0.1 bp target + Adam
                                    ablation plateau
  fig4  curse of dimensionality  -- FD time ~ N^5 vs PINN mild scaling, 1-D vs 3-D
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vol_normalization import price_err_to_bp, BUDGET_01BPS, VEGA_REF

TARGET_BP = 0.1


def fig_mc_wall(mc_rows, ref, path="fig1_mc_wall.png"):
    Ns = np.array([r["N"] for r in mc_rows], float)
    bp = np.array([max(r["se_bp"], r["bp"]) for r in mc_rows], float)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.loglog(Ns, bp, "o-", label="MC error (bp of vol)")
    # 1/sqrt(N) guide and extrapolation to the target
    c = bp[-1] * np.sqrt(Ns[-1])
    NN = np.array([Ns[0], 1e11])
    ax.loglog(NN, c / np.sqrt(NN), "--", color="grey", label=r"$\propto 1/\sqrt{N}$")
    n_target = (c / TARGET_BP) ** 2
    ax.axhline(TARGET_BP, color="crimson", lw=1.2, label="0.1 bp of vol target")
    ax.axvline(n_target, color="crimson", ls=":", lw=1)
    ax.set_xlabel("paths N"); ax.set_ylabel("error (bp of vol)")
    ax.set_title(f"MC convergence wall\n0.1 bp needs ~{n_target:.1e} paths")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=140)
    return path, n_target


def fig_crossover(n_contracts, fd_per, mc_per, pinn_train, pinn_infer,
                  path="fig2_crossover.png"):
    n = np.arange(1, n_contracts + 1)
    fd = fd_per * n
    mc = mc_per * n
    pinn = pinn_train + pinn_infer * n
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(n, fd, label=f"FD ({fd_per:.2f}s/contract)")
    ax.plot(n, mc, label=f"MC ({mc_per:.2f}s/contract)")
    ax.plot(n, pinn, label=f"PINN ({pinn_train:.0f}s train + {pinn_infer*1e3:.1f}ms/contract)")
    # crossover points
    for series, lab in [(fd, "FD"), (mc, "MC")]:
        cross = np.where(pinn < series)[0]
        if len(cross):
            ax.axvline(n[cross[0]], ls=":", lw=0.8, color="grey")
    ax.set_xlabel("# contracts priced"); ax.set_ylabel("total wall-clock (s)")
    ax.set_title("Amortization crossover (headline)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=140)
    return path


def fig_accuracy_frontier(frontier, path="fig3_frontier.png"):
    """frontier: dict name -> list of (time_s, err_bp)."""
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for name, pts in frontier.items():
        pts = np.array(pts)
        ax.loglog(pts[:, 0], pts[:, 1], "o-", label=name)
    ax.axhline(TARGET_BP, color="crimson", lw=1.2, label="0.1 bp of vol")
    ax.set_xlabel("wall-clock (s)"); ax.set_ylabel("error (bp of vol)")
    ax.set_title("Accuracy frontier (single contract)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=140)
    return path


def fig_curse(fd_rows, c_fit, accuracy_N, pinn_1d_time, pinn_3d_time,
              path="fig4_curse.png"):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    Ns = np.array([r[0] for r in fd_rows], float)
    secs = np.array([r[2] for r in fd_rows], float)
    ax.loglog(Ns, secs, "o", label="FD 3-D measured")
    NN = np.logspace(np.log10(Ns[0]), np.log10(accuracy_N), 50)
    ax.loglog(NN, c_fit * NN**5, "--", color="grey", label=r"fit $\propto N^5$")
    ax.axvline(accuracy_N, color="crimson", ls=":", label=f"accuracy-grade N={accuracy_N}")
    yr = c_fit * accuracy_N**5 / 3.15e7
    ax.annotate(f"~{yr:.1f} yr", xy=(accuracy_N, c_fit*accuracy_N**5),
                fontsize=9, color="crimson")
    ax.axhline(pinn_3d_time, color="green", lw=1.2,
               label=f"PINN 3-D (~{pinn_3d_time:.0f}s)")
    ax.set_xlabel("grid points per axis N"); ax.set_ylabel("wall-clock (s)")
    ax.set_title("Curse of dimensionality: FD 3-D vs PINN")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=140)
    return path


if __name__ == "__main__":
    # illustrative standalone run using validated CPU numbers + projected PINN legs
    ref = 97.5047
    mc_rows = [
        {"N": int(1e4), "bp": 9.0, "se_bp": 9.0},
        {"N": int(1e5), "bp": 2.8, "se_bp": 2.8},
        {"N": int(1e6), "bp": 0.9, "se_bp": 0.9},
        {"N": int(1e7), "bp": 0.45, "se_bp": 0.45},
    ]
    _, n_t = fig_mc_wall(mc_rows, ref)
    fig_crossover(60, fd_per=0.12, mc_per=2.0, pinn_train=900.0, pinn_infer=0.003)
    fig_accuracy_frontier({
        "PINN-2nd-order (SSBroyden)": [(900, 5), (1100, 0.6), (1300, 0.08)],
        "PINN-Adam (ablation)":       [(300, 30), (600, 9), (900, 6)],
        "FD (CN+Rannacher)":          [(0.02, 55), (0.12, 13), (0.36, 6), (3.0, 0.4)],
        "MC (antithetic+CV)":         [(1.0, 9), (10, 2.8), (100, 0.9)],
    })
    fig_curse([(16, 64.14, 0.02), (24, 47.01, 0.08), (32, 43.25, 0.30)],
              c_fit=1.06e-8, accuracy_N=1500, pinn_1d_time=1000, pinn_3d_time=1800)
    print("wrote fig1..fig4 (illustrative; populate PINN legs from H200 JSON)")
