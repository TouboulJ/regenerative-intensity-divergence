#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensite_hawkes.py  --  validation du Theoreme 3 sur un processus AUTO-EXCITANT.

Processus de Hawkes exponentiel : lambda(t)=mu + a*beta*sum_{t_i<t} exp(-beta (t-t_i)).
C'est le cas le plus exigeant couvert par le Theoreme 3 : l'intensite est
VRAIMENT predictible (depend des evenements passes), donc l'argument martingale
(CLT de Rebolledo) est indispensable -- on ne peut pas se ramener a de l'i.i.d.
(renouvellement) ni a une intensite deterministe (Poisson inhomogene).

On estime theta=(mu, a) (beta connu, fixe le timescale) par :
  - MLE          (= DPD a alpha=0)
  - DPD(alpha)   (Basu) : robuste, sans bande passante

Trois verifications :
  (1) EFFICACITE au modele                 (MSE relative, ref MLE)
  (2) ROBUSTESSE                           (evenements parasites injectes)
  (3) VARIANCE SANDWICH du Theoreme 3      (couverture des IC 95% + QQ-plot de
      normalite des estimateurs standardises par l'ecart-type sandwich plug-in)
  -> (3) est le test direct de la conclusion du Theoreme 3.

USAGE :  python intensite_hawkes.py   [--reps 200] [--T 100] [--alpha 0.5]
SORTIE : sorties_hawkes/{tables,figures} A COTE du script.
Dependances : numpy, scipy (requis) ; matplotlib (figure).
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, csv, argparse
import numpy as np
from scipy import optimize, stats

OUTPUT_DIRNAME = "sorties_hawkes"; RESULTS_SUBDIR = "tables"; FIGURE_SUBDIR = "figures"
def script_dir():
    try: return os.path.dirname(os.path.abspath(__file__))
    except NameError: return os.path.dirname(os.path.abspath(sys.argv[0])) or os.getcwd()

# ---------------- simulation (Ogata, somme courante O(N)) -------------------
def simulate_hawkes(mu, a, beta, T, rng):
    t = 0.0; ev = []; last = 0.0; S = 0.0
    while True:
        lam_bar = mu + a*beta*S
        w = rng.exponential(1.0/lam_bar); t = t + w
        if t >= T: break
        S = S*np.exp(-beta*(t-last)); last = t       # decroissance jusqu'a t
        lam_t = mu + a*beta*S
        if rng.uniform() <= lam_t/lam_bar:
            ev.append(t); S += 1.0                     # acceptation : ajoute le noyau
    return np.array(ev)

# ---------------- quantites pre-calculables (beta connu) --------------------
def event_kernels(ev, beta):
    """A_i = exp(-beta dt)(1+A_{i-1}) ; lambda(t_i)=mu+a beta A_i."""
    N = len(ev); A = np.zeros(N)
    for i in range(1, N):
        A[i] = np.exp(-beta*(ev[i]-ev[i-1]))*(1.0+A[i-1])
    return A

def grid_kernels(ev, beta, T, m=600):
    """K(g)=sum_{t_i<g} exp(-beta(g-t_i)) sur une grille ; independant de (mu,a)."""
    grid = np.linspace(0, T, m); dg = grid[1]-grid[0]
    K = np.zeros(m); S = 0.0; idx = 0
    for k in range(m):
        if k > 0: S *= np.exp(-beta*dg)
        while idx < len(ev) and ev[idx] <= grid[k]:
            S += np.exp(-beta*(grid[k]-ev[idx])); idx += 1
        K[k] = S
    return grid, dg, K

def tail_integral(ev, beta, T):
    return np.sum(1.0 - np.exp(-beta*(T-ev)))   # int des noyaux = a * cette somme

# ---------------- estimateurs (theta=(mu,a), beta connu) -------------------
def fit_mle(ev, beta, T, A, tail):
    def nll(th):
        mu, a = th
        li = mu + a*beta*A
        if np.any(li <= 0): return 1e12
        return mu*T + a*tail - np.sum(np.log(li))
    r = optimize.minimize(nll, [max(len(ev)/T*0.5,1e-2), 0.3], method="L-BFGS-B",
                          bounds=[(1e-3, None), (1e-3, 0.98)])
    return r.x

def fit_dpd(ev, beta, T, A, grid, dg, K, alpha):
    def obj(th):
        mu, a = th
        li = mu + a*beta*A
        lg = mu + a*beta*K
        if np.any(li <= 0) or np.any(lg <= 0): return 1e12
        return np.sum(lg**(1+alpha))*dg - (1+1/alpha)*np.sum(li**alpha)
    r = optimize.minimize(obj, [max(len(ev)/T*0.5,1e-2), 0.3], method="L-BFGS-B",
                          bounds=[(1e-3, None), (1e-3, 0.98)])
    return r.x

def sandwich_se(th, beta, grid, dg, K, alpha):
    """ECART-TYPE sandwich plug-in du Theoreme 3 : Var = J^{-1} Sigma J^{-1}."""
    mu, a = th; lg = mu + a*beta*K
    # score s(g)=(d log l/d mu, d log l/d a) = (1/l, beta K / l)
    s0 = 1.0/lg; s1 = beta*K/lg
    def mat(power):
        w = lg**power
        M = np.zeros((2,2))
        M[0,0]=np.sum(w*s0*s0)*dg; M[0,1]=M[1,0]=np.sum(w*s0*s1)*dg; M[1,1]=np.sum(w*s1*s1)*dg
        return M
    J = mat(alpha+1); Sig = mat(2*alpha+1)
    try:
        Jinv = np.linalg.inv(J); Var = Jinv@Sig@Jinv
        return np.sqrt(np.clip(np.diag(Var), 0, None))
    except np.linalg.LinAlgError:
        return np.array([np.nan, np.nan])

def contaminate(ev, T, eps, rng):
    m = int(eps*max(len(ev),1))
    if m <= 0: return ev
    spur = rng.uniform(0, T, m)              # evenements parasites (hors auto-excitation)
    return np.sort(np.concatenate([ev, spur]))

def gof_pvalue(ev, th, beta, A):
    """Goodness-of-fit par THEOREME DE RE-ECHELONNAGE (time-rescaling, Ogata).
    Sous le bon modele, les increments du compensateur Lambda(t_i) sont i.i.d.
    Exp(1). Test de Kolmogorov-Smirnov -> p-value (grande = modele compatible)."""
    mu, a = th; k = np.arange(len(ev))
    Lam = mu*ev + a*(k - A)                  # Lambda(t_k) = mu t_k + a (k - A_k)
    dL = np.diff(Lam); dL = dL[dL >= 0]      # increments ~ Exp(1) si correct
    if len(dL) < 10: return np.nan
    return float(stats.kstest(dL, 'expon').pvalue)

def gof_pvalue_poisson(ev, T):
    """MEME GOF mais sous un MAUVAIS modele (Poisson homogene ajuste aux donnees).
    Sur des donnees Hawkes (en rafales), les increments ne sont PAS Exp(1)
    -> p-value ecrasee vers 0 : montre que le test a de la PUISSANCE."""
    if len(ev) < 11: return np.nan
    lam = len(ev)/T
    dL = lam*np.diff(np.sort(ev))            # increments sous Poisson homogene
    return float(stats.kstest(dL, 'expon').pvalue)

# ---------------- experiences ----------------------------------------------
def run(cfg, resdir, figdir):
    mu0, a0, beta = 1.0, 0.5, 1.0; th0 = np.array([mu0, a0])
    T = cfg["T"]; reps = cfg["reps"]; alpha = cfg["alpha"]
    print(f"\nHawkes exp.  mu={mu0}, a={a0}, beta={beta}, T={T}, reps={reps}, alpha={alpha}")
    print(f"taux stationnaire ~ mu/(1-a) = {mu0/(1-a0):.2f}  => ~{int(mu0/(1-a0)*T)} evts/realisation")

    def one(eps, seed, collect_sandwich=False):
        rng = np.random.default_rng(seed)
        sq = {"MLE": [], "DPD(0.1)": [], f"DPD({alpha})": []}
        z_mle = []; cover = 0; ntot = 0; gof = []; gof_wrong = []
        for _ in range(reps):
            ev = simulate_hawkes(mu0, a0, beta, T, rng)
            if len(ev) < 15: continue
            evc = contaminate(ev, T, eps, rng) if eps > 0 else ev
            A = event_kernels(evc, beta); tail = tail_integral(evc, beta, T)
            grid, dg, K = grid_kernels(evc, beta, T)
            mle = fit_mle(evc, beta, T, A, tail)
            d1  = fit_dpd(evc, beta, T, A, grid, dg, K, 0.1)
            d5  = fit_dpd(evc, beta, T, A, grid, dg, K, alpha)
            sq["MLE"].append(np.sum((mle-th0)**2))
            sq["DPD(0.1)"].append(np.sum((d1-th0)**2))
            sq[f"DPD({alpha})"].append(np.sum((d5-th0)**2))
            g = gof_pvalue(evc, mle, beta, A)
            if np.isfinite(g): gof.append(g)
            gw = gof_pvalue_poisson(evc, T)
            if np.isfinite(gw): gof_wrong.append(gw)
            if collect_sandwich:
                se = sandwich_se(mle, beta, grid, dg, K, 0.0)  # MLE : J=Sigma=I
                if np.all(np.isfinite(se)) and np.all(se > 0):
                    zmu = (mle[0]-mu0)/se[0]; za = (mle[1]-a0)/se[1]
                    z_mle.append(zmu)
                    cover += int(abs(zmu) < 1.96) + int(abs(za) < 1.96); ntot += 2
        res = {k: float(np.mean(v)) for k, v in sq.items()}
        return res, np.array(z_mle), (cover, ntot), np.array(gof), np.array(gof_wrong)

    print("\n=== (1) EFFICACITE au modele (MSE, ref MLE) ===")
    m0, zret, (cov_c, cov_n), gof_ok, gof_wrong = one(0.0, 11, collect_sandwich=True)
    covg = cov_c/cov_n if cov_n else float("nan")
    eff = {k: m0["MLE"]/m0[k] for k in m0}
    for k in m0: print(f"  {k:12s} MSE={m0[k]:.5f}   eff={eff[k]:.2f}")
    rows_eff = [(k, m0[k], eff[k]) for k in m0]

    print(f"\n=== (2) ROBUSTESSE ({int(cfg['eps']*100)}% evenements parasites) ===")
    mc, _, _, _, _ = one(cfg["eps"], 22)
    gain = {k: mc["MLE"]/mc[k] for k in mc}
    for k in mc: print(f"  {k:12s} MSE={mc[k]:.5f}   gain={gain[k]:.2f}")
    rows_rob = [(k, mc[k], gain[k]) for k in mc]

    print("\n=== (3) VARIANCE SANDWICH (Theoreme 3) ===")
    print(f"  couverture IC 95% (MLE, sandwich plug-in) = {covg*100:.1f}%   (cible : 95%)")
    print(f"  normalite de (muhat-mu0)/SE : skew={stats.skew(zret):.2f}, "
          f"kurtosis(exces)={stats.kurtosis(zret):.2f}   (gaussien : 0,0)")

    print("\n=== (4) P-VALEURS (tests formels) ===")
    p_shapiro = float(stats.shapiro(zret).pvalue) if len(zret) >= 8 else float("nan")
    p_cover = float(stats.binomtest(cov_c, cov_n, 0.95).pvalue) if cov_n else float("nan")
    rej_ok = float(np.mean(gof_ok < 0.05)); rej_wrong = float(np.mean(gof_wrong < 0.05))
    print(f"  (a) normalite z (Shapiro-Wilk)        : p = {p_shapiro:.3f}   "
          f"(p>0.05 => compatible gaussien)")
    print(f"  (b) couverture = 95% (binomial exact) : p = {p_cover:.3f}   "
          f"(p>0.05 => couverture compatible 95%)")
    print(f"  (c) GOF time-rescaling (KS Exp(1)), distribution des p-values :")
    print(f"        bon modele (Hawkes)     : p moyen={np.mean(gof_ok):.3f}, "
          f"rejet a 5% = {rej_ok*100:.1f}%   (~Uniforme ; plug-in => un peu conservateur)")
    print(f"        mauvais modele (Poisson): p moyen={np.mean(gof_wrong):.3f}, "
          f"rejet a 5% = {rej_wrong*100:.1f}%   (eleve = le test a de la PUISSANCE)")

    rows_sw = [("couverture_IC95_pct", round(covg*100,2)),
               ("skew_z", round(float(stats.skew(zret)),3)),
               ("excess_kurtosis_z", round(float(stats.kurtosis(zret)),3)),
               ("p_normalite_shapiro", round(p_shapiro,4)),
               ("p_couverture_binomial", round(p_cover,4)),
               ("GOF_bon_modele_p_moyen", round(float(np.mean(gof_ok)),4)),
               ("GOF_bon_modele_rejet_pct", round(rej_ok*100,2)),
               ("GOF_mauvais_modele_p_moyen", round(float(np.mean(gof_wrong)),4)),
               ("GOF_mauvais_modele_rejet_pct", round(rej_wrong*100,2))]

    _csv(resdir, "efficacite.csv", ["method","MSE","eff_vs_MLE"], rows_eff)
    _csv(resdir, "robustesse.csv", ["method","MSE","gain_vs_MLE"], rows_rob)
    _csv(resdir, "sandwich.csv", ["quantite","valeur"], rows_sw)
    _figure(figdir, rows_eff, rows_rob, zret, covg, alpha, gof_ok, gof_wrong)

def _csv(resdir, name, header, rows):
    os.makedirs(resdir, exist_ok=True)
    with open(os.path.join(resdir, name), "w", newline="") as f:
        w = csv.writer(f); w.writerow(header)
        for r in rows: w.writerow(list(r))

def _figure(figdir, rows_eff, rows_rob, zret, covg, alpha, gof_ok, gof_wrong):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e: print(f"[info] figure ignoree ({e})"); return
    os.makedirs(figdir, exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8)); axs = axs.ravel()
    ms = [r[0] for r in rows_eff]
    cols = ['0.6' if m=='MLE' else ('tab:green' if '0.1' in m else 'tab:blue') for m in ms]
    axs[0].bar(range(len(ms)), [r[1] for r in rows_eff], color=cols)
    axs[0].set_xticks(range(len(ms))); axs[0].set_xticklabels(ms, fontsize=8, rotation=10)
    axs[0].set_ylabel("MSE"); axs[0].set_title("(1) Efficacite au modele (MSE, bas=mieux)", fontsize=9)
    axs[0].grid(axis='y', alpha=.3)
    axs[1].bar(range(len(ms)), [r[1] for r in rows_rob], color=cols)
    axs[1].set_xticks(range(len(ms))); axs[1].set_xticklabels(ms, fontsize=8, rotation=10)
    axs[1].set_ylabel("MSE"); axs[1].set_title("(2) Robustesse parasites (MSE, bas=mieux)", fontsize=9)
    axs[1].grid(axis='y', alpha=.3)
    # (3) QQ-plot de normalite
    zs = np.sort(zret); q = stats.norm.ppf((np.arange(1, len(zs)+1)-0.5)/len(zs))
    axs[2].plot(q, zs, 'o', color='tab:purple', ms=3, alpha=0.6)
    lim = [min(q.min(), zs.min()), max(q.max(), zs.max())]
    axs[2].plot(lim, lim, 'r-', lw=1)
    axs[2].set_xlabel("quantiles normaux"); axs[2].set_ylabel("(muhat-mu0)/SE_sandwich")
    axs[2].set_title(f"(3) Normalite + sandwich : couverture IC95% = {covg*100:.0f}%", fontsize=9)
    axs[2].grid(alpha=.3)
    # (4) GOF p-values (time-rescaling) : au modele ~ Uniforme ; contamine ~ pique a 0
    bins = np.linspace(0, 1, 21)
    axs[3].hist(gof_ok, bins=bins, density=True, alpha=0.6, color='tab:green',
                label='bon modele Hawkes (~ Uniforme)')
    axs[3].hist(gof_wrong, bins=bins, density=True, alpha=0.6, color='tab:red',
                label='mauvais modele Poisson (~ pique a 0)')
    axs[3].axhline(1.0, color='0.5', ls='--', lw=1)
    axs[3].set_xlabel("p-value GOF (KS Exp(1))"); axs[3].set_ylabel("densite")
    axs[3].set_title("(4) GOF time-rescaling : p-values", fontsize=9)
    axs[3].legend(fontsize=8); axs[3].grid(alpha=.3)
    fig.tight_layout(); p = os.path.join(figdir, "hawkes_benchmark.png"); fig.savefig(p, dpi=130); plt.close(fig)
    print(f"        {p}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--eps", type=float, default=0.15)
    ap.add_argument("--alpha", type=float, default=0.5)
    a = ap.parse_args(); cfg = dict(reps=a.reps, T=a.T, eps=a.eps, alpha=a.alpha)
    base = os.path.join(script_dir(), OUTPUT_DIRNAME)
    resdir = os.path.join(base, RESULTS_SUBDIR); figdir = os.path.join(base, FIGURE_SUBDIR)
    os.makedirs(resdir, exist_ok=True); os.makedirs(figdir, exist_ok=True)
    print(f"Sortie -> {base}")
    run(cfg, resdir, figdir)
    print(f"\nEcrit : {resdir}/  (csv)")

if __name__ == "__main__":
    main()
