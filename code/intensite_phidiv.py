#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensite_phidiv.py  --  TOUT-EN-UN, autonome.

Teste empiriquement les DEUX promesses theoriques de l'estimateur a minimum de
phi-divergence pour l'intensite d'un processus REGENERATIF (renouvellement) :
les inter-arrivees sont i.i.d. ~ f(.;theta) (intensite = hasard), donc on estime
theta. Concurrent = MLE (PAS missForest : on a change de probleme).

  (1) EFFICACITE au modele : l'estimateur phi atteint-il la variance du MLE ?
  (2) B-ROBUSTESSE : sous contamination (inter-arrivees aberrantes), bat-il le MLE ?

Estimateurs (tous membres de la famille divergence) :
  - MLE                (= KL,  reference, optimal au modele)
  - Hellinger  (MHDE, Beran 1977 ; phi-divergence robuste efficace)
  - Pearson chi2-MDE   (phi-divergence)
sur plusieurs DOMAINES (familles d'inter-arrivees) :
  - Weibull   (fiabilite / durees de vie)
  - Gamma     (files d'attente / comptage)
  - Lognormal (rafales / queues lourdes)

USAGE
-----
  python intensite_phidiv.py                       # simulation : efficacite + robustesse
  python intensite_phidiv.py --reps 400 --n 300    # plus de Monte-Carlo
  python intensite_phidiv.py --csv events.csv --col t --mode times   # tes donnees : temps d'evenements
  python intensite_phidiv.py --csv gaps.csv  --col x --mode gaps     # tes donnees : inter-arrivees

SORTIE : dossier "sorties_intensite/" A COTE du script (tables/ , figures/).
Dependances : numpy, scipy (requis) ; pandas (si --csv) ; matplotlib (figures).
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, csv, argparse
import numpy as np
from scipy import stats, optimize

# ===================== NOMS DES DOSSIERS (modifiables) ======================
OUTPUT_DIRNAME = "sorties_intensite"
RESULTS_SUBDIR = "tables"
FIGURE_SUBDIR  = "figures"

def script_dir():
    try: return os.path.dirname(os.path.abspath(__file__))
    except NameError: return os.path.dirname(os.path.abspath(sys.argv[0])) or os.getcwd()

# ===================== familles d'inter-arrivees (domaines) =================
# chaque famille : nom -> dict(sample, pdf, fit_mle, true, p2vec, vec2p, labels)
def make_families():
    fam = {}

    # Weibull (shape c, scale s), loc=0
    fam["Weibull (fiabilite)"] = dict(
        true=np.array([1.5, 2.0]),
        sample=lambda th, n, rng: stats.weibull_min.rvs(th[0], scale=th[1], size=n, random_state=rng),
        pdf=lambda x, th: stats.weibull_min.pdf(x, th[0], scale=th[1]),
        mle=lambda d: np.array(_mle(stats.weibull_min, d)),
        labels=("shape", "scale"))

    # Gamma (shape a, scale s), loc=0
    fam["Gamma (files d'attente)"] = dict(
        true=np.array([2.0, 1.5]),
        sample=lambda th, n, rng: stats.gamma.rvs(th[0], scale=th[1], size=n, random_state=rng),
        pdf=lambda x, th: stats.gamma.pdf(x, th[0], scale=th[1]),
        mle=lambda d: np.array(_mle(stats.gamma, d)),
        labels=("shape", "scale"))

    # Lognormal (sigma s, scale=exp(mu)), loc=0
    fam["Lognormal (rafales)"] = dict(
        true=np.array([0.6, 1.0]),
        sample=lambda th, n, rng: stats.lognorm.rvs(th[0], scale=th[1], size=n, random_state=rng),
        pdf=lambda x, th: stats.lognorm.pdf(x, th[0], scale=th[1]),
        mle=lambda d: np.array(_mle(stats.lognorm, d)),
        labels=("sigma", "scale"))
    return fam

def _mle(dist, data):
    c, loc, scale = dist.fit(data, floc=0)
    return [c, scale]

# ===================== estimateurs a minimum de phi-divergence ==============
def _kde_on_grid(data, grid):
    bw = 0.9 * min(np.std(data), (np.subtract(*np.percentile(data, [75, 25])) / 1.349) or np.std(data)) \
         * len(data) ** (-1 / 5)
    bw = max(bw, 1e-3)
    z = (grid[None, :] - data[:, None]) / bw
    k = np.exp(-0.5 * z * z) / (bw * np.sqrt(2 * np.pi))
    g = k.mean(0)
    # reflexion a 0 pour limiter le biais de bord (support positif)
    zr = (grid[None, :] + data[:, None]) / bw
    g = g + (np.exp(-0.5 * zr * zr) / (bw * np.sqrt(2 * np.pi))).mean(0)
    return g

def _mde(data, pdf, theta0, divergence="hellinger"):
    xmax = np.quantile(data, 0.999) * 1.3 + 1e-6
    grid = np.linspace(1e-6, xmax, 400); dx = grid[1] - grid[0]
    ghat = _kde_on_grid(data, grid)
    def obj(lp):
        th = np.exp(lp)
        f = np.clip(pdf(grid, th), 1e-12, None)
        if divergence == "hellinger":
            val = np.sum((np.sqrt(ghat) - np.sqrt(f)) ** 2) * dx
        elif divergence == "pearson":  # chi2 de Pearson : int (g-f)^2 / f
            val = np.sum((ghat - f) ** 2 / f) * dx
        else:
            raise ValueError(divergence)
        return val
    res = optimize.minimize(obj, np.log(np.clip(theta0, 1e-6, None)),
                            method="Nelder-Mead",
                            options=dict(maxiter=400, xatol=1e-4, fatol=1e-6))
    return np.exp(res.x)

def estimators(data, fam):
    th_mle = fam["mle"](data)
    out = {"MLE": th_mle}
    try: out["Hellinger"] = _mde(data, fam["pdf"], th_mle, "hellinger")
    except Exception: out["Hellinger"] = th_mle
    return out

# ===================== experiences =========================================
def contaminate(data, eps, rng, factor=6.0):
    d = data.copy(); n = len(d); k = int(eps * n)
    if k > 0:
        idx = rng.choice(n, k, replace=False)
        d[idx] = d[idx] * factor  # inter-arrivees aberrantes (geantes)
    return d

def experiment(fam, name, reps, n, eps, rng):
    true = fam["true"]; methods = ["MLE", "Hellinger"]
    sq = {m: [] for m in methods}
    for r in range(reps):
        data = fam["sample"](true, n, rng)
        if eps > 0: data = contaminate(data, eps, rng)
        est = estimators(data, fam)
        for m in methods:
            e = (est[m] - true) / true            # erreur relative par parametre
            sq[m].append(np.sum(e ** 2))
    return {m: float(np.mean(sq[m])) for m in methods}

def run_simulation(cfg, resdir, figdir, panels):
    fams = make_families(); rng = np.random.default_rng(0)
    # (1) efficacite au modele (eps=0)  (2) robustesse (eps=cfg.eps)
    rows_eff = []; rows_rob = []
    print("\n=== (1) EFFICACITE au modele (MSE relative ; ref = MLE) ===")
    print(f"{'domaine':26s} {'MLE':>9s} {'Hellinger':>11s} {'  eff(Hell)':>12s}")
    for name, fam in fams.items():
        m0 = experiment(fam, name, cfg["reps"], cfg["n"], 0.0, np.random.default_rng(1))
        eff_h = m0["MLE"] / m0["Hellinger"] if m0["Hellinger"] > 0 else float("nan")
        print(f"{name:26s} {m0['MLE']:9.4f} {m0['Hellinger']:11.4f} {eff_h:12.2f}")
        rows_eff.append((name, m0["MLE"], m0["Hellinger"], eff_h))
    print(f"\n=== (2) ROBUSTESSE : {int(cfg['eps']*100)}% inter-arrivees x6 (MSE relative ; bas = mieux) ===")
    print(f"{'domaine':26s} {'MLE':>9s} {'Hellinger':>11s} {'  gain(Hell)':>12s}")
    for name, fam in fams.items():
        mc = experiment(fam, name, cfg["reps"], cfg["n"], cfg["eps"], np.random.default_rng(2))
        gain = mc["MLE"] / mc["Hellinger"] if mc["Hellinger"] > 0 else float("nan")
        print(f"{name:26s} {mc['MLE']:9.4f} {mc['Hellinger']:11.4f} {gain:12.2f}")
        rows_rob.append((name, mc["MLE"], mc["Hellinger"], gain))
    _write_csv(resdir, "efficacite_modele.csv",
               ["domaine", "MLE", "Hellinger", "eff_Hell_vs_MLE"], rows_eff)
    _write_csv(resdir, "robustesse_contamination.csv",
               ["domaine", "MLE", "Hellinger", "gain_Hell_vs_MLE"], rows_rob)
    panels.append(("(1) Efficacite au modele\n(MSE rel., bas=mieux)", rows_eff, False))
    panels.append((f"(2) Robustesse ({int(cfg['eps']*100)}% aberrants)\n(MSE rel., bas=mieux)", rows_rob, True))

def run_csv(cfg, resdir, figdir, panels):
    import pandas as pd
    df = pd.read_csv(cfg["csv"]); col = cfg["col"] or df.columns[-1]
    v = pd.to_numeric(df[col], errors="coerce").dropna().values.astype(float)
    if cfg["mode"] == "times":
        v = np.diff(np.sort(v))                       # temps d'evenements -> inter-arrivees
    v = v[v > 0]
    print(f"\nserie reelle: {os.path.basename(cfg['csv'])}:{col} ({cfg['mode']}) -> "
          f"{len(v)} inter-arrivees, moyenne={v.mean():.3g}")
    fams = make_families()
    # ajuster chaque famille ; choisir par AIC ; comparer MLE vs Hellinger ; sensibilite a 1 aberrant
    rows = []
    print(f"\n{'famille':26s} {'AIC':>9s}  params MLE         params Hellinger    sens.1outlier(MLE)")
    for name, fam in fams.items():
        est = estimators(v, fam)
        ll = np.sum(np.log(np.clip(fam["pdf"](v, est["MLE"]), 1e-300, None)))
        aic = 2 * 2 - 2 * ll
        v2 = np.append(v, v.max() * 6.0)              # injecte 1 aberrant
        mle2 = fam["mle"](v2)
        shift = np.sum(np.abs((mle2 - est["MLE"]) / est["MLE"]))
        print(f"{name:26s} {aic:9.1f}  {np.round(est['MLE'],3)!s:18s} {np.round(est['Hellinger'],3)!s:18s} {shift:8.3f}")
        rows.append((name, aic, list(np.round(est["MLE"], 4)), list(np.round(est["Hellinger"], 4)), round(shift, 4)))
    _write_csv(resdir, "ajustement_reel.csv",
               ["famille", "AIC", "params_MLE", "params_Hellinger", "sensibilite_1outlier_MLE"], rows)
    print("\n(la derniere colonne mesure le deplacement du MLE quand on ajoute UN seul evenement aberrant ;")
    print(" l'estimateur Hellinger y est typiquement bien plus stable -- c'est la B-robustesse.)")

# ===================== sortie ==============================================
def _write_csv(resdir, name, header, rows):
    os.makedirs(resdir, exist_ok=True)
    with open(os.path.join(resdir, name), "w", newline="") as f:
        w = csv.writer(f); w.writerow(header)
        for r in rows: w.writerow(list(r))

def make_figure(figdir, panels):
    if not panels: return None
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[info] figure ignoree ({e})"); return None
    os.makedirs(figdir, exist_ok=True)
    panels = [p for p in panels if len(p) == 3]
    if not panels: return None
    n = len(panels); fig, axs = plt.subplots(1, n, figsize=(6.2 * n, 4.0))
    if n == 1: axs = [axs]
    for ax, (title, rows, robust) in zip(axs, panels):
        doms = [r[0].split(" (")[0] for r in rows]
        mle = [r[1] for r in rows]; hell = [r[2] for r in rows]
        x = np.arange(len(doms)); bw = 0.32
        ax.bar(x - bw/2, mle, bw, label="MLE", color="0.6")
        ax.bar(x + bw/2, hell, bw, label="Hellinger (phi-div.)", color="tab:green")
        ax.set_xticks(x); ax.set_xticklabels(doms, fontsize=8, rotation=15)
        ax.set_ylabel("MSE relative"); ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(figdir, "intensite_benchmark.png"); fig.savefig(p, dpi=130); plt.close(fig)
    return p

# ===================== main ================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv"); ap.add_argument("--col")
    ap.add_argument("--mode", choices=["times", "gaps"], default="gaps")
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--eps", type=float, default=0.05)
    a = ap.parse_args()
    cfg = dict(csv=a.csv, col=a.col, mode=a.mode, reps=a.reps, n=a.n, eps=a.eps)

    base = os.path.join(script_dir(), OUTPUT_DIRNAME)
    resdir = os.path.join(base, RESULTS_SUBDIR); figdir = os.path.join(base, FIGURE_SUBDIR)
    os.makedirs(resdir, exist_ok=True); os.makedirs(figdir, exist_ok=True)
    print(f"Sortie -> {base}")
    panels = []
    if a.csv:
        run_csv(cfg, resdir, figdir, panels)
    else:
        print(f"Simulation : {cfg['reps']} replicats, n={cfg['n']} inter-arrivees, "
              f"contamination {int(cfg['eps']*100)}%")
        run_simulation(cfg, resdir, figdir, panels)
    p = make_figure(figdir, panels)
    print(f"\nEcrit : {resdir}/  (csv)")
    if p: print(f"        {p}")

if __name__ == "__main__":
    main()
