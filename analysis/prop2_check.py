"""THEORY.md Proposition 2 check (no API calls).

Log-odds increment of an agreed-upon cluster c from (a) a new author vs (b) a repeat whose earlier answers all lie
in c. Computed directly with mathmage.team.posterior (difficulty fixed through a tiny b_sd; no traps) and
cross-checked against the closed form in the paper's appendix proof. Positive differences mean the repeat is the
weaker evidence.

    python analysis/prop2_check.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mathmage.team.posterior import AgentProfile, Obs, PosteriorParams, TeamPosterior  # noqa: E402


def closed(p, kappa, kappa_same, rho, W=1, n=1):
    o, os_ = kappa / (1 - kappa), kappa_same / (1 - kappa_same)
    za = 1 + W * o
    col_a, col_b = np.log(za / o), np.log((za - o + os_) / os_)
    cor_a = np.log(p / (1 - p))
    if rho == 0:
        cor_b = cor_a
    else:
        nu = (1 - rho) / rho
        cor_b = np.log((nu * p + n) / (nu * (1 - p) + n))
    return (cor_a + col_a) - (cor_b + col_b)


def direct(p, kappa, kappa_same, rho):
    theta = float(np.log(p / (1 - p)))
    agents = {k: AgentProfile(theta=theta, abstain=0.0) for k in ("A", "B", "X")}
    params = PosteriorParams(agents=agents, b_mean=0.0, b_sd=1e-6, kappa=kappa, kappa_same=kappa_same,
                             trap_weight=0.0, kappa_trap=0.0, same_agent_correlation=rho, grid=3)
    tp = TeamPosterior(params)
    base = [Obs("X", 0), Obs("A", 1)]          # cluster 0 (another answer) by X, cluster 1 by A

    def logodds(obs):
        r = tp.compute(obs, n_clusters=2)
        return np.log(r.q[1] / r.q[0])

    l0 = logodds(base)
    return (logodds(base + [Obs("B", 1)]) - l0) - (logodds(base + [Obs("A", 1)]) - l0)


# Frozen fits: runs/live_20261018/program_v3/offline_sat and program_open/offline_open artifacts.
TEAMS = {"SAT": (0.033027, 0.220815, 0.195538), "open": (0.054215, 0.054215, 0.238078)}

if __name__ == "__main__":
    for name, (k, ks, rho) in TEAMS.items():
        print(f"\n{name}: kappa={k} kappa_same={ks} rho={rho}")
        for p in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
            print(f"  p={p:.2f}  closed {closed(p, k, ks, rho):+.3f}  direct {direct(p, k, ks, rho):+.3f}  "
                  f"rho=0: closed {closed(p, k, ks, 0.0):+.3f}  direct {direct(p, k, ks, 0.0):+.3f}")
        ps = np.linspace(0.001, 0.999, 9981)
        d = np.array([closed(p, k, ks, rho) for p in ps])
        neg = ps[d < 0]
        print("  repeat is stronger evidence than a new author for p <",
              round(float(neg.max()), 3) if len(neg) else "never")
