#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensite_nonrenouvellement.py  --  cas NON-renouvellement (la vraie nouveaute).

Processus de Poisson INHOMOGENE d'intensite lambda(t,theta)=exp(b0+b1 z(t)),
evenements non i.i.d. en temps. On estime theta=(b0,b1) par :
  - MLE                         : maximise  sum log lambda(tau_i) - int lambda dt
  - Hellinger-intensite (MHDI)  : minimise  int (sqrt(ghat_h) - sqrt(lambda))^2 dt
                                  (ghat_h = intensite a noyau)  -> efficace 1er ordre + robuste
  - DPD(alpha) (Basu)           : minimise  int lambda^(1+a) dt - (1+1/a) sum lambda(tau_i)^a
                                  -> robuste, efficacite asymptotique < 1 (prix de la robustesse)

Trois experiences :
  (1) EFFICACITE au modele        (MSE relative, ref MLE)
  (2) ROBUSTESSE                  (evenements parasites injectes)
  (3) EFFICACITE vs T             (MHDI -> 1 ? DPD -> plateau ?)

USAGE :  python intensite_nonrenouvellement.py            [--reps 200] [--alpha 0.5]
SORTIE : sorties_nonrenouv/{tables,figures} A COTE du script.
Dependances : numpy, scipy (requis) ; matplotlib (figure).
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, csv, argparse
import numpy as np
from scipy import optimize

OUTPUT_DIRNAME = "sorties_nonrenouv"; RESULTS_SUBDIR = "tables"; FIGURE_SUBDIR = "figures"
def script_dir():
    try: return os.path.dirname(os.path.abspath(__file__))
    except NameError: return os.path.dirname(os.path.abspath(sys.argv[0])) or os.getcwd()

# ---------------- modele : lambda(t)=exp(b0+b1 z(t)), z(t)=sin(2 pi t/P) -----
def zfun(t, T): return np.sin(2*np.pi*t/(T/3.0))
def lam(t, th, T): return np.exp(th[0] + th[1]*zfun(t, T))

def simulate_ipp(th, T, rng):
    """Poisson inhomogene par amincissement (Lewis-Shedler)."""
    lo = th[0] - abs(th[1]); hi = th[0] + abs(th[1]); lmax = np.exp(hi)
    n = rng.poisson(lmax*T); cand = np.sort(rng.uniform(0, T, n))
    keep = rng.uniform(0, 1, n) < lam(cand, th, T)/lmax
    return cand[keep]

# ---------------- estimateurs ----------------------------------------------
def _grid(T, m=600):
    g = np.linspace(0, T, m); return g, g[1]-g[0]

def nll_mle(th, ev, T, grid, dg):
    return np.sum(lam(grid, th, T))*dg - np.sum(np.log(np.clip(lam(ev, th, T), 1e-300, None)))

def fit_mle(ev, T, th0, grid, dg):
    r = optimize.minimize(nll_mle, th0, args=(ev, T, grid, dg), method="Nelder-Mead",
                          options=dict(maxiter=600, xatol=1e-4, fatol=1e-6))
    return r.x

def _kernel_intensity(ev, grid, h):
    z = (grid[None, :] - ev[:, None]) / h
    return (np.exp(-0.5*z*z)/(h*np.sqrt(2*np.pi))).sum(0)

def fit_mhdi(ev, T, th0, grid, dg):
    """Hellinger sur l'intensite : min int (sqrt(ghat)-sqrt(lambda))^2 dt."""
    if len(ev) < 5: return th0
    h = 0.9*np.std(ev)*len(ev)**(-1/5); h = max(h, T/100.0)
    ghat = _kernel_intensity(ev, grid, h)
    sg = np.sqrt(np.clip(ghat, 0, None))
    def obj(th):
        sl = np.sqrt(np.clip(lam(grid, th, T), 1e-300, None))
        return np.sum((sg - sl)**2)*dg
    r = optimize.minimize(obj, th0, method="Nelder-Mead",
                          options=dict(maxiter=600, xatol=1e-4, fatol=1e-6))
    return r.x

def fit_dpd(ev, T, th0, grid, dg, alpha=0.5):
    def obj(th):
        l = np.clip(lam(grid, th, T), 1e-300, None)
        le = np.clip(lam(ev, th, T), 1e-300, None)
        return np.sum(l**(1+alpha))*dg - (1+1/alpha)*np.sum(le**alpha)
    r = optimize.minimize(obj, th0, method="Nelder-Mead",
                          options=dict(maxiter=600, xatol=1e-4, fatol=1e-6))
    return r.x

def estimate_all(ev, T, alpha):
    grid, dg = _grid(T); th0 = np.array([np.log(max(len(ev),1)/T), 0.0])
    mle = fit_mle(ev, T, th0, grid, dg)
    return {"MLE": mle,
            "DPD(0.1)": fit_dpd(ev, T, mle, grid, dg, 0.1),
            "DPD(0.5)": fit_dpd(ev, T, mle, grid, dg, 0.5)}

def contaminate(ev, T, eps, rng):
    """Parasites dans les CREUX d'intensite (la ou le modele dit 'peu d'evenements')."""
    m = int(eps*max(len(ev),1))
    if m <= 0: return ev
    P = T/3.0; troughs = [0.75*P + k*P for k in range(3) if 0.75*P + k*P < T]
    centers = rng.choice(troughs, m); spur = np.clip(centers + rng.normal(0, P*0.03, m), 0, T)
    return np.sort(np.concatenate([ev, spur]))

# ---------------- experiences ----------------------------------------------
def mse_over_reps(th_true, T, reps, eps, alpha, seed):
    rng = np.random.default_rng(seed)
    methods = ["MLE", "DPD(0.1)", "DPD(0.5)"]
    sq = {m: [] for m in methods}
    for _ in range(reps):
        ev = simulate_ipp(th_true, T, rng)
        if len(ev) < 8: continue
        ec = contaminate(ev, T, eps, rng) if eps > 0 else ev
        est = estimate_all(ec, T, alpha)
        for m in methods:
            sq[m].append(np.sum(((est[m]-th_true))**2))
    return {m: float(np.mean(sq[m])) for m in methods}

def run(cfg, resdir, figdir, panels):
    th_true = np.array([np.log(8.0), 1.0])   # ~ moderee
    T0 = cfg["T"]; a = cfg["alpha"]
    print(f"\nPoisson inhomogene  lambda=exp({th_true[0]:.2f}+{th_true[1]:.2f} sin), "
          f"T={T0}, reps={cfg['reps']}, alpha={a}")

    print("\n=== (1) EFFICACITE au modele (MSE rel., ref MLE) ===")
    m0 = mse_over_reps(th_true, T0, cfg["reps"], 0.0, a, 11)
    eff = {k: m0["MLE"]/m0[k] for k in m0}
    for k in m0: print(f"  {k:14s} MSE={m0[k]:.5f}   eff={eff[k]:.2f}")
    rows_eff = [(k, m0[k], eff[k]) for k in m0]

    print(f"\n=== (2) ROBUSTESSE ({int(cfg['eps']*100)}% evenements parasites en rafale) ===")
    mc = mse_over_reps(th_true, T0, cfg["reps"], cfg["eps"], a, 22)
    gain = {k: mc["MLE"]/mc[k] for k in mc}
    for k in mc: print(f"  {k:14s} MSE={mc[k]:.5f}   gain={gain[k]:.2f}")
    rows_rob = [(k, mc[k], gain[k]) for k in mc]

    print("\n=== (3) EFFICACITE vs T (au modele) ===")
    Ts = [T0, 2*T0, 4*T0, 8*T0]; curve = {"DPD(0.1)": [], "DPD(0.5)": []}
    print(f"  {'T':>7s} {'eff DPD(0.1)':>13s} {'eff DPD(0.5)':>13s}")
    rows_curve = []
    for T in Ts:
        mt = mse_over_reps(th_true, T, max(cfg["reps"]//2, 80), 0.0, a, 33)
        e1 = mt["MLE"]/mt["DPD(0.1)"]; e5 = mt["MLE"]/mt["DPD(0.5)"]
        curve["DPD(0.1)"].append(e1); curve["DPD(0.5)"].append(e5)
        print(f"  {T:7d} {e1:13.2f} {e5:13.2f}"); rows_curve.append((T, e1, e5))

    _csv(resdir, "efficacite_modele.csv", ["method","MSE","eff_vs_MLE"], rows_eff)
    _csv(resdir, "robustesse.csv", ["method","MSE","gain_vs_MLE"], rows_rob)
    _csv(resdir, "efficacite_vs_T.csv", ["T","eff_DPD01","eff_DPD05"], rows_curve)
    panels.append(("eff", rows_eff, rows_rob, Ts, curve, a))

def _csv(resdir, name, header, rows):
    os.makedirs(resdir, exist_ok=True)
    with open(os.path.join(resdir, name), "w", newline="") as f:
        w = csv.writer(f); w.writerow(header)
        for r in rows: w.writerow(list(r))

def make_figure(figdir, panels):
    if not panels: return None
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e: print(f"[info] figure ignoree ({e})"); return None
    os.makedirs(figdir, exist_ok=True)
    _, rows_eff, rows_rob, Ts, curve, a = panels[0]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    # (1) efficacite
    ms=[r[0] for r in rows_eff]; mse=[r[1] for r in rows_eff]
    cols=['0.6' if m=='MLE' else ('tab:green' if '0.1' in m else 'tab:blue') for m in ms]
    axs[0].bar(range(len(ms)), mse, color=cols); axs[0].set_xticks(range(len(ms)))
    axs[0].set_xticklabels(ms, fontsize=8, rotation=10); axs[0].set_ylabel("MSE"); 
    axs[0].set_title("(1) Efficacite au modele\n(MSE, bas=mieux)", fontsize=9); axs[0].grid(axis='y',alpha=.3)
    # (2) robustesse
    msr=[r[1] for r in rows_rob]
    axs[1].bar(range(len(ms)), msr, color=cols); axs[1].set_xticks(range(len(ms)))
    axs[1].set_xticklabels(ms, fontsize=8, rotation=10); axs[1].set_ylabel("MSE")
    axs[1].set_title("(2) Robustesse (parasites)\n(MSE, bas=mieux)", fontsize=9); axs[1].grid(axis='y',alpha=.3)
    # (3) eff vs T
    axs[2].axhline(1.0, color='0.5', ls='--', lw=1, label='efficacite 1.0 (MLE)')
    axs[2].plot(Ts, curve["DPD(0.1)"], '-o', color='tab:green', label='DPD(0.1)')
    axs[2].plot(Ts, curve["DPD(0.5)"], '-s', color='tab:blue', label='DPD(0.5)')
    axs[2].set_xlabel("T"); axs[2].set_ylabel("efficacite (MSE_MLE/MSE)")
    axs[2].set_title("(3) Efficacite vs T", fontsize=9); axs[2].legend(fontsize=8); axs[2].grid(alpha=.3)
    fig.tight_layout(); p=os.path.join(figdir,"nonrenouv_benchmark.png"); fig.savefig(p,dpi=130); plt.close(fig)
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--T", type=int, default=60)
    ap.add_argument("--eps", type=float, default=0.10)
    ap.add_argument("--alpha", type=float, default=0.5)
    a = ap.parse_args()
    cfg = dict(reps=a.reps, T=a.T, eps=a.eps, alpha=a.alpha)
    base = os.path.join(script_dir(), OUTPUT_DIRNAME)
    resdir = os.path.join(base, RESULTS_SUBDIR); figdir = os.path.join(base, FIGURE_SUBDIR)
    os.makedirs(resdir, exist_ok=True); os.makedirs(figdir, exist_ok=True)
    print(f"Sortie -> {base}")
    panels = []; run(cfg, resdir, figdir, panels)
    p = make_figure(figdir, panels)
    print(f"\nEcrit : {resdir}/  (csv)")
    if p: print(f"        {p}")

if __name__ == "__main__":
    main()
