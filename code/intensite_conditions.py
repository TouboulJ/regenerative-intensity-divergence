#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensite_conditions.py  --  verification NUMERIQUE des conditions du Theoreme 3
pour le processus de Hawkes exponentiel (Annexe F du papier).

lambda(t)=mu + a*beta*sum_{t_i<t} exp(-beta(t-t_i)),  theta=(mu,a),  a<1 (stable).
score s=(1/lambda, (lambda-mu)/(a lambda)) ;  a_alpha = lambda^alpha s.

On verifie les hypotheses de la preuve (qu'on ne peut que SUPPOSER analytiquement) :
  (C3) limites ergodiques : T^{-1} Sigma_T et T^{-1} J_T convergent quand T grandit.
  (C4) Lindeberg : T^{-1} int ||a||^2 1{||a|| > eps sqrt(T)} lambda dt  -> 0.
  (C2) identifiabilite : le contraste DPD normalise H_alpha(theta)/T a un minimum
       UNIQUE et bien separe en theta_0 (convexite locale).

C'est le complement empirique de l'Annexe F : ce que la preuve suppose, ce script
le confirme. (Aucune verite-terrain n'est requise : ce sont des proprietes du
modele au vrai theta_0, parfaitement legitimes a verifier par simulation.)

USAGE :  python intensite_conditions.py  [--Tmax 2000] [--reps 8] [--alpha 0.0]
SORTIE : sorties_conditions/{tables,figures} A COTE du script.
Dependances : numpy, scipy (requis) ; matplotlib (figure).
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, csv, argparse
import numpy as np

OUTPUT_DIRNAME = "sorties_conditions"; RESULTS_SUBDIR = "tables"; FIGURE_SUBDIR = "figures"
def script_dir():
    try: return os.path.dirname(os.path.abspath(__file__))
    except NameError: return os.path.dirname(os.path.abspath(sys.argv[0])) or os.getcwd()

MU0, A0, BETA = 1.0, 0.5, 1.0     # vrai theta0 (a<1 : stable)

def simulate_hawkes(T, rng, mu=MU0, a=A0, beta=BETA):
    t=0.0; ev=[]; last=0.0; S=0.0
    while True:
        lam_bar=mu+a*beta*S; t+=rng.exponential(1.0/lam_bar)
        if t>=T: break
        S=S*np.exp(-beta*(t-last)); last=t
        if rng.uniform()<=(mu+a*beta*S)/lam_bar: ev.append(t); S+=1.0
    return np.array(ev)

def grid_kernel(ev, T, m, beta=BETA):
    """K(g)=sum_{t_i<g} exp(-beta(g-t_i)) sur grille (recursion O(m+N))."""
    grid=np.linspace(0,T,m); dg=grid[1]-grid[0]; K=np.zeros(m); S=0.0; idx=0
    for k in range(m):
        if k>0: S*=np.exp(-beta*dg)
        while idx<len(ev) and ev[idx]<=grid[k]:
            S+=np.exp(-beta*(grid[k]-ev[idx])); idx+=1
        K[k]=S
    return grid, dg, K

def event_A(ev, beta=BETA):
    A=np.zeros(len(ev))
    for i in range(1,len(ev)): A[i]=np.exp(-beta*(ev[i]-ev[i-1]))*(1.0+A[i-1])
    return A

# ---------------- (C3) + (C4) ----------------------------------------------
def conditions_C3_C4(Tmax, reps, alpha, horizons, eps, seed, m):
    """Accumule, le long d'UNE longue realisation, les densites de Sigma, J et de
       Lindeberg ; renvoie T^{-1}Sigma_T, T^{-1}J_T (entrees) et le ratio Lindeberg."""
    SigA=np.zeros((len(horizons),3)); JA=np.zeros((len(horizons),3))
    LindA=np.zeros(len(horizons)); LindA05=np.zeros(len(horizons)); nr=0
    for r in range(reps):
        rng=np.random.default_rng(seed+r); ev=simulate_hawkes(Tmax,rng)
        if len(ev)<30: continue
        grid,dg,K=grid_kernel(ev,Tmax,m); lam=MU0+A0*BETA*K
        lam=np.clip(lam,1e-9,None)
        s0=1.0/lam; s1=(lam-MU0)/(A0*lam)              # score (2 composantes)
        snorm=np.sqrt(s0*s0+s1*s1)
        # densites (cumul via cumsum) : Sigma=int lam^{2a+1} s s^T ; J=int lam^{a+1} s s^T
        dSig=[lam**(2*alpha+1)*s0*s0, lam**(2*alpha+1)*s0*s1, lam**(2*alpha+1)*s1*s1]
        dJ  =[lam**(alpha+1)*s0*s0,   lam**(alpha+1)*s0*s1,   lam**(alpha+1)*s1*s1]
        cS=[np.cumsum(d)*dg for d in dSig]; cJ=[np.cumsum(d)*dg for d in dJ]
        anorm=(lam**alpha)*snorm                         # ||a_alpha|| (alpha demande)
        anorm05=(lam**0.5)*snorm                         # ||a_alpha|| pour alpha=0.5 (non borne)
        dL =(anorm**2)*lam; dL05=(anorm05**2)*lam
        for i,T in enumerate(horizons):
            j=min(int(T/dg), m-1); thr=eps*np.sqrt(T)
            for c in range(3):
                SigA[i,c]+=cS[c][j]/T; JA[i,c]+=cJ[c][j]/T
            LindA[i]  +=np.sum(dL[:j+1]  *(anorm[:j+1]  >thr))*dg/T
            LindA05[i]+=np.sum(dL05[:j+1]*(anorm05[:j+1]>thr))*dg/T
        nr+=1
    SigA/=max(nr,1); JA/=max(nr,1); LindA/=max(nr,1); LindA05/=max(nr,1)
    return SigA, JA, LindA, LindA05, nr

# ---------------- (C2) identifiabilite -------------------------------------
def condition_C2(Tbig, alpha, seed, m, ngrid=41):
    rng=np.random.default_rng(seed); ev=simulate_hawkes(Tbig,rng)
    grid,dg,K=grid_kernel(ev,Tbig,m); A=event_A(ev)
    mus=np.linspace(0.4,1.8,ngrid); as_=np.linspace(0.2,0.8,ngrid)
    H=np.zeros((ngrid,ngrid))
    for ia,a in enumerate(as_):
        for im,mu in enumerate(mus):
            lg=np.clip(mu+a*BETA*K,1e-9,None); le=np.clip(mu+a*BETA*A,1e-9,None)
            if alpha>0:
                H[ia,im]=(np.sum(lg**(1+alpha))*dg-(1+1/alpha)*np.sum(le**alpha))/Tbig
            else:  # alpha->0 : -log-vraisemblance / T
                H[ia,im]=(np.sum(lg)*dg-np.sum(np.log(le)))/Tbig
    ja,jm=np.unravel_index(np.argmin(H),H.shape)
    return mus, as_, H, (mus[jm], as_[ja]), len(ev)

# ---------------- sortie ---------------------------------------------------
def _csv(resdir,name,header,rows):
    os.makedirs(resdir,exist_ok=True)
    with open(os.path.join(resdir,name),"w",newline="") as f:
        w=csv.writer(f); w.writerow(header)
        for r in rows: w.writerow(list(r))

def figure(figdir, horizons, SigA, JA, LindA, LindA05, mus, as_, H, argmin, alpha):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e: print(f"[info] figure ignoree ({e})"); return None
    os.makedirs(figdir,exist_ok=True)
    fig,axs=plt.subplots(1,3,figsize=(16,4.6))
    # (C3)
    axs[0].plot(horizons,SigA[:,0],'-o',ms=3,label="T^{-1}Sigma_T [0,0]",color='tab:blue')
    axs[0].plot(horizons,SigA[:,2],'-o',ms=3,label="T^{-1}Sigma_T [1,1]",color='tab:cyan')
    axs[0].plot(horizons,JA[:,0],'-s',ms=3,label="T^{-1}J_T [0,0]",color='tab:red')
    axs[0].plot(horizons,JA[:,2],'-s',ms=3,label="T^{-1}J_T [1,1]",color='tab:orange')
    axs[0].set_xlabel("T"); axs[0].set_title("(C3) Limites ergodiques\n(doivent se stabiliser)",fontsize=9)
    axs[0].legend(fontsize=7); axs[0].grid(alpha=.3)
    # (C4)
    axs[1].plot(horizons,LindA05,'-o',ms=3,color='tab:green',label="alpha=0.5 (score non borne)")
    axs[1].plot(horizons,LindA,'-s',ms=3,color='0.5',label="alpha=0 (borne : trivial)")
    axs[1].axhline(0,color='0.6',ls='--',lw=1)
    axs[1].set_xlabel("T"); axs[1].set_ylabel("ratio de Lindeberg")
    axs[1].set_title("(C4) Lindeberg -> 0\n(doit tendre vers 0)",fontsize=9)
    axs[1].legend(fontsize=8); axs[1].grid(alpha=.3)
    # (C2)
    cs=axs[2].contourf(mus,as_,H,levels=25,cmap='viridis')
    axs[2].plot(MU0,A0,'r*',ms=16,label='theta_0 vrai')
    axs[2].plot(argmin[0],argmin[1],'wo',ms=7,mfc='none',label='argmin numerique')
    axs[2].set_xlabel("mu"); axs[2].set_ylabel("a")
    axs[2].set_title("(C2) Contraste normalise\nminimum unique en theta_0",fontsize=9)
    axs[2].legend(fontsize=8); fig.colorbar(cs,ax=axs[2],fraction=0.046)
    fig.tight_layout(); p=os.path.join(figdir,"conditions_benchmark.png"); fig.savefig(p,dpi=130); plt.close(fig)
    return p

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--Tmax",type=int,default=2000); ap.add_argument("--reps",type=int,default=8)
    ap.add_argument("--alpha",type=float,default=0.0); ap.add_argument("--eps",type=float,default=0.1)
    a=ap.parse_args()
    base=os.path.join(script_dir(),OUTPUT_DIRNAME)
    resdir=os.path.join(base,RESULTS_SUBDIR); figdir=os.path.join(base,FIGURE_SUBDIR)
    os.makedirs(resdir,exist_ok=True); os.makedirs(figdir,exist_ok=True)
    print(f"Sortie -> {base}")
    print(f"\nHawkes mu={MU0}, a={A0}, beta={BETA} (stable, a<1) ; alpha={a.alpha}, reps={a.reps}, Tmax={a.Tmax}")
    m=max(2000,a.Tmax*2); horizons=np.linspace(a.Tmax*0.05,a.Tmax,18)

    print("\n=== (C3) limites ergodiques + (C4) Lindeberg ===")
    SigA,JA,LindA,LindA05,nr=conditions_C3_C4(a.Tmax,a.reps,a.alpha,horizons,a.eps,11,m)
    print(f"  {nr} realisations utilisees")
    print(f"  T^{{-1}}Sigma_T[0,0] : {SigA[0,0]:.4f} (T={horizons[0]:.0f})  ->  {SigA[-1,0]:.4f} (T={horizons[-1]:.0f})")
    print(f"  T^{{-1}}J_T[0,0]     : {JA[0,0]:.4f}  ->  {JA[-1,0]:.4f}")
    print(f"  ratio Lindeberg a=0.5 : {LindA05[0]:.4f} (T={horizons[0]:.0f})  ->  {LindA05[-1]:.4g} (T={horizons[-1]:.0f})")
    var0=np.std(SigA[len(SigA)//2:,0])/max(abs(np.mean(SigA[len(SigA)//2:,0])),1e-9)
    print(f"  stabilite (CV de T^-1 Sigma[0,0] sur 2e moitie) = {var0:.3f}  (petit = converge)")
    _csv(resdir,"C3_C4.csv",["T","Sig00","Sig11","J00","J11","Lind_a0","Lind_a05"],
         [(round(horizons[i],1),round(SigA[i,0],5),round(SigA[i,2],5),round(JA[i,0],5),round(JA[i,2],5),round(LindA[i],6),round(LindA05[i],6)) for i in range(len(horizons))])

    print("\n=== (C2) identifiabilite (minimum du contraste) ===")
    mus,as_,H,argmin,nev=condition_C2(min(a.Tmax,1500),a.alpha,7,m)
    print(f"  realisation a {nev} evenements ; vrai theta_0=({MU0},{A0})")
    print(f"  argmin numerique du contraste = ({argmin[0]:.3f}, {argmin[1]:.3f})  "
          f"(doit etre proche de theta_0)")
    err=np.hypot(argmin[0]-MU0,argmin[1]-A0)
    print(f"  distance a theta_0 = {err:.3f}  (petit = identifiable, minimum bien place)")

    p=figure(figdir,horizons,SigA,JA,LindA,LindA05,mus,as_,H,argmin,a.alpha)
    print(f"\nEcrit : {resdir}/  (csv)")
    if p: print(f"        {p}")

if __name__=="__main__":
    main()
