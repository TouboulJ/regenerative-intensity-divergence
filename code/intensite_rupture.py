#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensite_rupture.py  --  rupture dans l'intensite (Theoreme 2 / Conjecture 2).

Poisson constant par morceaux sur [0,1] : lambda = n*rho1 si t<tau0, n*rho2 sinon
(rho1 != rho2). On estime la localisation tau0 par maximum de vraisemblance
profilee. Theorie de Kutoyants/Ibragimov-Khasminskii :
  - le rate est n (l'echelle d'intensite), PAS sqrt(n) ;
  - n*(tauhat - tau0) converge en loi vers l'argmax d'un champ a sauts (Poisson
    bilateral) : limite NON gaussienne.

Le script verifie empiriquement (1) le rate via le scaling de la MSE, et
(2) la non-gaussianite de la loi limite (histogramme + asymetrie/kurtosis).

USAGE :  python intensite_rupture.py  [--reps 400]
SORTIE : sorties_rupture/{tables,figures} A COTE du script.
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, csv, argparse
import numpy as np
from scipy import stats

OUTPUT_DIRNAME = "sorties_rupture"; RESULTS_SUBDIR = "tables"; FIGURE_SUBDIR = "figures"
def script_dir():
    try: return os.path.dirname(os.path.abspath(__file__))
    except NameError: return os.path.dirname(os.path.abspath(sys.argv[0])) or os.getcwd()

def simulate(n, rho1, rho2, tau0, rng):
    """Poisson constant par morceaux sur [0,1], echelle n."""
    n1 = rng.poisson(n*rho1*tau0); n2 = rng.poisson(n*rho2*(1-tau0))
    e1 = rng.uniform(0, tau0, n1); e2 = rng.uniform(tau0, 1, n2)
    return np.sort(np.concatenate([e1, e2]))

def estimate_tau(ev):
    """argmax de la log-vraisemblance profilee sur les splits candidats."""
    N = len(ev)
    if N < 4: return 0.5
    # candidats = milieux entre evenements consecutifs
    cand = (ev[:-1] + ev[1:]) / 2.0
    best = None
    for tau in cand:
        N1 = np.searchsorted(ev, tau); N2 = N - N1
        if N1 < 1 or N2 < 1: continue
        # profil : N1 log(N1/tau) + N2 log(N2/(1-tau))  (constantes omises)
        ll = N1*np.log(N1/tau) + N2*np.log(N2/(1-tau))
        if best is None or ll > best[1]: best = (tau, ll)
    return best[0] if best else 0.5

def run(cfg, resdir, figdir):
    rho1, rho2, tau0 = 1.0, 3.0, 0.5
    ns = [50, 100, 200, 400, 800]; reps = cfg["reps"]
    print(f"\nRupture-en-intensite : [0,1], rho1={rho1}, rho2={rho2}, tau0={tau0}")
    print(f"{'n':>6s} {'MSE(tau)':>12s} {'n*MSE':>10s} {'n^2*MSE':>12s}")
    rows = []
    for n in ns:
        rng = np.random.default_rng(100+n); errs = []
        for _ in range(reps):
            ev = simulate(n, rho1, rho2, tau0, rng)
            errs.append(estimate_tau(ev) - tau0)
        errs = np.array(errs); mse = float(np.mean(errs**2))
        print(f"{n:6d} {mse:12.3e} {n*mse:10.4f} {n*n*mse:12.4f}")
        rows.append((n, mse, n*mse, n*n*mse))
    # loi limite a grand n : echantillon de n*(tauhat-tau0)
    nbig = 800; rng = np.random.default_rng(7)
    scaled = np.array([nbig*(estimate_tau(simulate(nbig, rho1, rho2, tau0, rng)) - tau0)
                       for _ in range(max(reps, 600))])
    sk = float(stats.skew(scaled)); ku = float(stats.kurtosis(scaled))  # exces de kurtosis
    print(f"\nLoi limite de n*(tauhat-tau0) a n={nbig} :  "
          f"skew={sk:.2f}  kurtosis(exces)={ku:.2f}  (gaussien: 0, 0)")

    _csv(resdir, "rate_scaling.csv", ["n","MSE","n_MSE","n2_MSE"], rows)
    _csv(resdir, "loi_limite_moments.csv", ["n","skew","excess_kurtosis"], [(nbig, sk, ku)])
    _figure(figdir, ns, rows, scaled, sk, ku)
    # diagnostic rate : si n^2*MSE ~ const -> rate n ; si n*MSE ~ const -> rate sqrt(n)
    n2 = np.array([r[3] for r in rows]); n1 = np.array([r[2] for r in rows])
    cv2 = n2.std()/n2.mean(); cv1 = n1.std()/n1.mean()
    print(f"\nDiagnostic rate : CV(n^2*MSE)={cv2:.2f}  vs  CV(n*MSE)={cv1:.2f}")
    print("  -> le plus stable (CV bas) indique le bon rate.",
          "rate n" if cv2 < cv1 else "rate sqrt(n)")

def _csv(resdir, name, header, rows):
    os.makedirs(resdir, exist_ok=True)
    with open(os.path.join(resdir, name), "w", newline="") as f:
        w = csv.writer(f); w.writerow(header)
        for r in rows: w.writerow(list(r))

def _figure(figdir, ns, rows, scaled, sk, ku):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e: print(f"[info] figure ignoree ({e})"); return
    os.makedirs(figdir, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    n1 = [r[2] for r in rows]; n2 = [r[3] for r in rows]
    axs[0].plot(ns, np.array(n1)/n1[0], '-s', color='tab:orange', label='n * MSE (normalise)')
    axs[0].plot(ns, np.array(n2)/n2[0], '-o', color='tab:green', label='n^2 * MSE (normalise)')
    axs[0].set_xlabel("n (echelle d'intensite)"); axs[0].set_ylabel("quantite normalisee")
    axs[0].set_title("Verification du rate : la courbe plate = bon rate", fontsize=9)
    axs[0].axhline(1.0, color='0.6', ls='--', lw=1); axs[0].legend(fontsize=8); axs[0].grid(alpha=.3)
    axs[1].hist(scaled, bins=40, density=True, color='tab:blue', alpha=0.7)
    mu, sd = scaled.mean(), scaled.std()
    xs = np.linspace(scaled.min(), scaled.max(), 200)
    axs[1].plot(xs, stats.norm.pdf(xs, mu, sd), 'r-', lw=1.5, label='gaussienne ajustee')
    axs[1].set_title(f"Loi limite de n(tauhat-tau0)\nskew={sk:.2f}, kurt={ku:.2f} (non gaussien)", fontsize=9)
    axs[1].set_xlabel("n (tauhat - tau0)"); axs[1].legend(fontsize=8); axs[1].grid(alpha=.3)
    fig.tight_layout(); p = os.path.join(figdir, "rupture_benchmark.png"); fig.savefig(p, dpi=130); plt.close(fig)
    print(f"        {p}")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=400)
    a = ap.parse_args(); cfg = dict(reps=a.reps)
    base = os.path.join(script_dir(), OUTPUT_DIRNAME)
    resdir = os.path.join(base, RESULTS_SUBDIR); figdir = os.path.join(base, FIGURE_SUBDIR)
    os.makedirs(resdir, exist_ok=True); os.makedirs(figdir, exist_ok=True)
    print(f"Sortie -> {base}")
    run(cfg, resdir, figdir)
    print(f"\nEcrit : {resdir}/  (csv)")

if __name__ == "__main__":
    main()
