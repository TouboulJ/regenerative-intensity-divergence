#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensite_conjecture1.py  --  test empirique de la CONJECTURE 1 (corrigee).

Correction (Hampel) : on NE PEUT PAS avoir efficacite=1 ET influence bornee a la
fois. La conjecture se scinde en deux faits demontrables, testes ici par SIMULATION
sur un processus de RENOUVELLEMENT (inter-arrivees i.i.d. ~ Gamma), le cadre ou le
score est non borne et ou vit l'estimateur de Hellinger de Beran :

  Theoreme A (Hellinger/Beran) : efficace au 1er ordre (eff -> 1) MAIS la
     condition est un SOUS-LISSAGE du noyau (h ~ n^{-1/3}, pas Silverman n^{-1/5}).
     Avec Silverman, le biais domine a l'echelle sqrt(n) et l'efficacite CHUTE.
  Theoreme B (Hampel/DPD)      : influence bornee, efficacite < 1 fixe (= f(alpha)),
     ne tend PAS vers 1.

  (A)  EFFICACITE vs n : MHDE sous-lisse -> ~1 ; MHDE Silverman -> chute ; DPD plateau.
  (B)  ROBUSTESSE (contamination) : a n fixe, MHDE et DPD battent le MLE.

IMPORTANT : l'efficacite ne se mesure QUE par simulation (il faut une verite-terrain).
Sur donnees reelles (--csv) on ne teste donc PAS l'efficacite : seulement
l'ajustement, le GOF (KS Gamma) et la sensibilite a un aberrant injecte.

USAGE :
  python intensite_conjecture1.py                         # simulation (A)+(B)
  python intensite_conjecture1.py --csv gaps.csv --col x --mode gaps   # inter-arrivees reelles
  python intensite_conjecture1.py --csv events.csv --col t --mode times
SORTIE : sorties_conjecture1/{tables,figures} A COTE du script.
Dependances : numpy, scipy (requis) ; pandas (si --csv) ; matplotlib (figure).
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, csv, argparse
import numpy as np
from scipy import stats, optimize

OUTPUT_DIRNAME = "sorties_conjecture1"; RESULTS_SUBDIR = "tables"; FIGURE_SUBDIR = "figures"
def script_dir():
    try: return os.path.dirname(os.path.abspath(__file__))
    except NameError: return os.path.dirname(os.path.abspath(sys.argv[0])) or os.getcwd()

TRUE = np.array([2.0, 1.5])   # Gamma(shape, scale) de reference

def mle(d):
    k, loc, sc = stats.gamma.fit(d, floc=0); return np.array([k, sc])

def _bw(d, rule):
    scale = min(np.std(d), (np.subtract(*np.percentile(d,[75,25]))/1.349) or np.std(d))
    n = len(d)
    if rule == "undersmooth": h = scale * n**(-1/3.0)   # Beran : n h^4 -> 0
    else:                      h = 0.9 * scale * n**(-1/5.0)  # Silverman
    return max(h, 1e-3)

def _kde(d, grid, h):
    z = (grid[None,:]-d[:,None])/h
    g = (np.exp(-0.5*z*z)/(h*np.sqrt(2*np.pi))).mean(0)
    zr = (grid[None,:]+d[:,None])/h
    return g + (np.exp(-0.5*zr*zr)/(h*np.sqrt(2*np.pi))).mean(0)

def mhde(d, th0, rule="undersmooth"):
    if len(d) < 8: return th0
    xmax = np.quantile(d,0.999)*1.3+1e-6; grid = np.linspace(1e-6,xmax,400); dx = grid[1]-grid[0]
    sg = np.sqrt(np.clip(_kde(d, grid, _bw(d, rule)), 0, None))
    def obj(lp):
        th = np.exp(lp); f = np.clip(stats.gamma.pdf(grid,th[0],scale=th[1]),1e-12,None)
        return np.sum((sg-np.sqrt(f))**2)*dx
    return np.exp(optimize.minimize(obj, np.log(np.clip(th0,1e-6,None)), method="Nelder-Mead",
                 options=dict(maxiter=400,xatol=1e-4,fatol=1e-6)).x)

def dpd(d, th0, a=0.5):
    xmax = np.quantile(d,0.999)*1.3+1e-6; grid = np.linspace(1e-6,xmax,400); dx = grid[1]-grid[0]
    def obj(lp):
        th = np.exp(lp); fg = np.clip(stats.gamma.pdf(grid,th[0],scale=th[1]),1e-300,None)
        fe = np.clip(stats.gamma.pdf(d,th[0],scale=th[1]),1e-300,None)
        return np.sum(fg**(1+a))*dx - (1+1/a)*np.mean(fe**a)
    return np.exp(optimize.minimize(obj, np.log(np.clip(th0,1e-6,None)), method="Nelder-Mead",
                 options=dict(maxiter=400,xatol=1e-4,fatol=1e-6)).x)

# ---------------- (A) efficacite vs n --------------------------------------
def exp_efficiency(ns, reps, alpha, seed):
    methods = ["MHDE (sous-lisse)", "MHDE (Silverman)", f"DPD({alpha})"]
    curve = {m: [] for m in methods}
    for n in ns:
        rng = np.random.default_rng(seed+n)
        sm=[]; sh=[]; ss=[]; sd=[]
        for _ in range(reps):
            d = stats.gamma.rvs(TRUE[0], scale=TRUE[1], size=n, random_state=rng)
            m = mle(d); sm.append(np.sum((m-TRUE)**2))
            sh.append(np.sum((mhde(d,m,"undersmooth")-TRUE)**2))
            ss.append(np.sum((mhde(d,m,"silverman")-TRUE)**2))
            sd.append(np.sum((dpd(d,m,alpha)-TRUE)**2))
        em = np.mean(sm)
        curve["MHDE (sous-lisse)"].append(em/np.mean(sh))
        curve["MHDE (Silverman)"].append(em/np.mean(ss))
        curve[f"DPD({alpha})"].append(em/np.mean(sd))
    return methods, curve

# ---------------- (B) robustesse a n fixe ----------------------------------
def exp_robustness(n, reps, alpha, eps, seed):
    methods = ["MLE", "MHDE (sous-lisse)", f"DPD({alpha})"]
    rng = np.random.default_rng(seed); sq = {m: [] for m in methods}
    for _ in range(reps):
        d = stats.gamma.rvs(TRUE[0], scale=TRUE[1], size=n, random_state=rng)
        k = int(eps*n)
        if k>0:
            idx = rng.choice(n,k,replace=False); d=d.copy(); d[idx]=d[idx]*6.0  # aberrants geants
        m = mle(d)
        sq["MLE"].append(np.sum((m-TRUE)**2))
        sq["MHDE (sous-lisse)"].append(np.sum((mhde(d,m,"undersmooth")-TRUE)**2))
        sq[f"DPD({alpha})"].append(np.sum((dpd(d,m,alpha)-TRUE)**2))
    return {m: float(np.mean(v)) for m,v in sq.items()}

# ---------------- donnees reelles ------------------------------------------
def run_csv(cfg, resdir):
    import pandas as pd
    df = pd.read_csv(cfg["csv"]); col = cfg["col"] or df.columns[-1]
    v = pd.to_numeric(df[col], errors="coerce").dropna().values.astype(float)
    x = np.diff(np.sort(v)) if cfg["mode"]=="times" else v
    x = x[x>0]
    print(f"\nDONNEES REELLES {os.path.basename(cfg['csv'])}:{col} -> {len(x)} inter-arrivees")
    print("ATTENTION : pas de verite-terrain => efficacite NON mesurable ici.")
    print("On verifie : ajustement, GOF (KS Gamma) et sensibilite a un aberrant.\n")
    m = mle(x); h = mhde(x,m,"undersmooth"); d = dpd(x,m,0.5)
    x_out = np.append(x, x.max()*6.0)   # 1 aberrant
    rows=[]; print(f"{'methode':18s} {'shape':>8s} {'scale':>8s} {'GOF p (KS)':>11s} {'sens.1outlier':>14s}")
    for name, th in [("MLE",m),("MHDE (sous-lisse)",h),("DPD(0.5)",d)]:
        p = float(stats.kstest(x, 'gamma', args=(th[0],0,th[1])).pvalue)
        if name=="MLE": th2=mle(x_out)
        elif name.startswith("MHDE"): th2=mhde(x_out,mle(x_out),"undersmooth")
        else: th2=dpd(x_out,mle(x_out),0.5)
        shift=float(np.linalg.norm(th2-th))
        print(f"{name:18s} {th[0]:8.3f} {th[1]:8.3f} {p:11.3f} {shift:14.4f}")
        rows.append((name,round(th[0],4),round(th[1],4),round(p,4),round(shift,4)))
    _csv(resdir,"reel_ajustement.csv",["methode","shape","scale","GOF_p_KS","sens_1outlier"],rows)
    print("\n(GOF p : KS des donnees vs Gamma ajuste -- conservateur, parametres estimes ;")
    print(" sens.1outlier : petit = robuste, MLE attendu le plus fragile.)")

# ---------------- sortie ---------------------------------------------------
def _csv(resdir, name, header, rows):
    os.makedirs(resdir, exist_ok=True)
    with open(os.path.join(resdir,name),"w",newline="") as f:
        w=csv.writer(f); w.writerow(header)
        for r in rows: w.writerow(list(r))

def make_figure(figdir, ns, mA, curve, rob, alpha):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e: print(f"[info] figure ignoree ({e})"); return None
    os.makedirs(figdir, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    colmap={"MHDE (sous-lisse)":"tab:green","MHDE (Silverman)":"tab:orange",f"DPD({alpha})":"tab:blue"}
    for m in mA:
        axs[0].plot(ns, curve[m], '-o', label=m, color=colmap.get(m,"0.4"))
    axs[0].axhline(1.0,color='0.6',ls='--',lw=1,label="efficacite 1 (MLE)")
    axs[0].set_xscale('log'); axs[0].set_xlabel("n (echelle log)"); axs[0].set_ylabel("efficacite (MSE_MLE/MSE)")
    axs[0].set_title("(A) Efficacite vs n\nHellinger sous-lisse -> 1 (Th.A) ; DPD plateau (Th.B) ;\nHellinger Silverman chute (fragilite bande passante)", fontsize=9)
    axs[0].legend(fontsize=8); axs[0].grid(alpha=.3)
    ms=list(rob.keys()); cols=['0.6' if m=='MLE' else ('tab:green' if 'MHDE' in m else 'tab:blue') for m in ms]
    axs[1].bar(range(len(ms)),[rob[m] for m in ms],color=cols)
    axs[1].set_xticks(range(len(ms))); axs[1].set_xticklabels(ms,fontsize=8,rotation=12)
    axs[1].set_ylabel("MSE"); axs[1].set_title("(B) Robustesse (contamination)\nMSE, bas=mieux",fontsize=9)
    axs[1].grid(axis='y',alpha=.3)
    fig.tight_layout(); p=os.path.join(figdir,"conjecture1_benchmark.png"); fig.savefig(p,dpi=130); plt.close(fig)
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv"); ap.add_argument("--col"); ap.add_argument("--mode",choices=["times","gaps"],default="gaps")
    ap.add_argument("--reps", type=int, default=120); ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--eps", type=float, default=0.05)
    a = ap.parse_args()
    base = os.path.join(script_dir(), OUTPUT_DIRNAME)
    resdir = os.path.join(base, RESULTS_SUBDIR); figdir = os.path.join(base, FIGURE_SUBDIR)
    os.makedirs(resdir,exist_ok=True); os.makedirs(figdir,exist_ok=True)
    print(f"Sortie -> {base}")
    if a.csv:
        run_csv(dict(csv=a.csv,col=a.col,mode=a.mode), resdir); return
    ns=[100,200,400,800,1600]
    print(f"\nRenouvellement Gamma(shape={TRUE[0]},scale={TRUE[1]}), alpha={a.alpha}, reps={a.reps}")
    print("\n=== (A) EFFICACITE vs n (ref MLE) ===")
    mA, curve = exp_efficiency(ns, a.reps, a.alpha, 1)
    print(f"  {'n':>6s} " + " ".join(f"{m:>18s}" for m in mA))
    rowsA=[]
    for i,n in enumerate(ns):
        print(f"  {n:6d} " + " ".join(f"{curve[m][i]:18.2f}" for m in mA))
        rowsA.append((n,*[round(curve[m][i],3) for m in mA]))
    _csv(resdir,"efficacite_vs_n.csv",["n",*mA],rowsA)
    print(f"\n=== (B) ROBUSTESSE ({int(a.eps*100)}% aberrants x6, n=400) ===")
    rob = exp_robustness(400, a.reps, a.alpha, a.eps, 7)
    for m in rob: print(f"  {m:18s} MSE={rob[m]:.5f}   gain vs MLE={rob['MLE']/rob[m]:.2f}")
    _csv(resdir,"robustesse.csv",["methode","MSE","gain_vs_MLE"],
         [(m,round(rob[m],6),round(rob['MLE']/rob[m],3)) for m in rob])
    p = make_figure(figdir, ns, mA, curve, rob, a.alpha)
    print(f"\nEcrit : {resdir}/  (csv)")
    if p: print(f"        {p}")

if __name__ == "__main__":
    main()
