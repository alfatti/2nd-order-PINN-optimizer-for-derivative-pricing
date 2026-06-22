"""
config.py
=========
Single source of truth for the Deng-Mallett-McCann (2011) discrete monthly
autocallable example, plus race / training settings.

Reference value (this package's validated ground truth): V0 ~ 97.51 per 100 face.
  * quadrature (reference_1d.py)         -> 97.505 (floor-converged)
  * CN+Rannacher FD (fd_1d.py)           -> 97.47-97.52 (grid-snapped)
  * antithetic MC 2e6 (mc_1d.py)         -> 97.511 +/- 0.006
  * par benchmark cross-check            -> 100.00 (validates payoff+discount)
The paper quotes 98.39 from a coarse EXPLICIT FD; that number carries
discretization bias from the payoff discontinuity at L and the call barrier and
is NOT used as ground truth here.
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class AutocallParams:
    S0:  float = 100.0      # initial reference price = face I
    I:   float = 100.0      # face value
    C:   float = 102.0      # call price (barrier)
    r:   float = 0.05       # risk-free
    sig: float = 0.20       # vol
    q:   float = 0.01       # dividend yield
    CDS: float = 0.01       # issuer CDS spread
    T:   float = 1.0        # maturity (yr)
    L:   float = 80.0       # downside protection threshold
    H:   float = 100.0      # called payoff scale  P_t = H e^{B t}
    B:   float = 0.092      # called annualized return
    n_calls: int = 12       # monthly; final call date == maturity

    @property
    def disc(self) -> float:
        return self.r + self.CDS

    @property
    def call_times(self) -> np.ndarray:
        return np.array([(i + 1) * self.T / self.n_calls
                         for i in range(self.n_calls)])

    def called_payoff(self, t):
        return self.H * np.exp(self.B * np.asarray(t))

    def maturity_payoff(self, S):
        """f(S): face if above threshold, else linear loss (discontinuous at L)."""
        S = np.asarray(S)
        return np.where(S > self.L, self.I, S)


# ---- worst-of-3 basket extension (curse-of-dimensionality demo) ----
@dataclass(frozen=True)
class WoF3Params:
    S0:   float = 100.0
    C:    float = 102.0
    r:    float = 0.05
    q:    float = 0.01
    CDS:  float = 0.01
    T:    float = 1.0
    L:    float = 80.0
    H:    float = 100.0
    B:    float = 0.092
    n_calls: int = 12
    vols: tuple = (0.20, 0.22, 0.25)
    # constant pairwise correlation
    rho:  float = 0.40

    @property
    def disc(self):
        return self.r + self.CDS

    @property
    def call_times(self):
        return np.array([(i + 1) * self.T / self.n_calls
                         for i in range(self.n_calls)])

    @property
    def corr(self):
        d = len(self.vols)
        M = np.full((d, d), self.rho)
        np.fill_diagonal(M, 1.0)
        return M

    @property
    def chol(self):
        return np.linalg.cholesky(self.corr)


# ---- vol-error normalization (identical convention to the Cheyette setup) ----
# A single fixed reference Normal Vega applied across all contracts, so the
# accuracy target is the same scalar everywhere rather than per-contract.
@dataclass(frozen=True)
class RaceConfig:
    bp_target: float = 0.1            # headline accuracy line: 0.1 bp of vol
    vega_ref: float = None            # set by vol_normalization.compute_vega_ref()
    seed: int = 0

    # PINN training (H200)
    pinn_width: int = 128
    pinn_depth: int = 5
    n_collocation: int = 200_000
    adam_steps: int = 20_000
    secondorder_steps: int = 3_000    # SSBroyden / SSBFGS outer iters

    # FD / MC reference settings
    fd_NX: int = 1600
    fd_NT: int = 1200
    mc_paths: int = 2_000_000

    # contract grid for the amortization crossover plot
    grid_C: tuple = (100.0, 101.0, 102.0, 103.0, 104.0)
    grid_T: tuple = (0.5, 0.75, 1.0, 1.5, 2.0)
    grid_sig: tuple = (0.15, 0.20, 0.25, 0.30)


PAR = AutocallParams()
WOF3 = WoF3Params()
RACE = RaceConfig()
