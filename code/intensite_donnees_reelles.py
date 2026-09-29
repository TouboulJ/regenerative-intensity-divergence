#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensite_donnees_reelles.py  --  application a un jeu REEL a EPOQUES OBSERVEES.

C'est le cas ou la theorie s'applique directement (epoques enregistrees, pas
latentes). On telecharge un catalogue d'evenements dates et on ajuste l'intensite
par les estimateurs du papier, avec goodness-of-fit. Deux jeux reels, deux
mecanismes :

  --jeu seismes   : catalogue USGS (auto-excitant -> Hawkes ; repliques)
  --jeu mines     : catastrophes minieres GB 1851-1962 (Poisson inhomogene ; 191 ev.)
  --csv fichier   : tes propres temps d'evenements (colonne --col, --mode times/gaps)

Analyses :
  (1) RENOUVELLEMENT : loi des inter-arrivees ajustee par MLE / MHDE(sous-lisse) /
      DPD(0.5) ; GOF par KS ; AIC ; sensibilite a un aberrant.
  (2) HAWKES (auto-excitation) : intensite mu+a*beta*sum exp(-beta .) ajustee par
      MLE et DPD ; GOF par re-echelonnage du temps (KS Exp(1)).

IMPORTANT : pas de verite-terrain sur du reel => on ne mesure PAS l'efficacite.
On rapporte ajustement, GOF et robustesse (sensibilite a un aberrant). C'est
exactement ce qu'on peut affirmer honnetement sur donnees reelles.

SORTIE : sorties_reelles/{tables,figures,donnees} A COTE du script.
Dependances : numpy, scipy, pandas (requis) ; matplotlib (figure). urllib (standard).
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, csv, argparse, urllib.request
import numpy as np
from scipy import stats, optimize

OUTPUT_DIRNAME="sorties_reelles"; RESULTS_SUBDIR="tables"; FIGURE_SUBDIR="figures"; DATA_SUBDIR="donnees"
def script_dir():
    try: return os.path.dirname(os.path.abspath(__file__))
    except NameError: return os.path.dirname(os.path.abspath(sys.argv[0])) or os.getcwd()

# ---------------- catalogue de jeux reels ----------------------------------
def usgs_url(start="2024-01-01", end="2024-12-31", minmag=4.5, bbox=None):
    u=("https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv"
       f"&starttime={start}&endtime={end}&minmagnitude={minmag}&orderby=time-asc")
    if bbox:  # minlat,maxlat,minlon,maxlon : isole une sequence regionale (1 noyau Hawkes)
        la0,la1,lo0,lo1=bbox
        u+=f"&minlatitude={la0}&maxlatitude={la1}&minlongitude={lo0}&maxlongitude={lo1}"
    return u
CATALOG = {
    "seismes": dict(url=usgs_url(), col="time", unit="days",
                    note="USGS seismes M>=4.5, 2024 (auto-excitant)"),
    "mines":   dict(url="https://vincentarelbundock.github.io/Rdatasets/csv/boot/coal.csv",
                    col="date", unit="years",
                    note="catastrophes minieres GB 1851-1962 (Poisson inhomogene)"),
}

def download(url, dest):
    if os.path.exists(dest):
        print(f"[cache] {dest}"); return dest
    print(f"[telechargement] {url}")
    try:
        req=urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r: data=r.read()
        with open(dest,"wb") as f: f.write(data)
        print(f"[ok] {dest} ({len(data)} octets)"); return dest
    except Exception as e:
        print(f"[warn] echec telechargement ({e})"); return None

def load_event_times(jeu, csv_path, col, mode, datadir, seismes_url=None, suffix=""):
    import pandas as pd
    if csv_path:
        df=pd.read_csv(csv_path); c=col or df.columns[-1]
        v=pd.to_numeric(df[c], errors="coerce").dropna().values.astype(float)
        ev=np.sort(v) if mode=="times" else np.cumsum(np.sort(v))
        return ev-ev.min(), f"{os.path.basename(csv_path)}:{c}"
    spec=CATALOG[jeu]; dest=os.path.join(datadir, jeu+suffix+".csv")
    url=seismes_url if (jeu=="seismes" and seismes_url) else spec["url"]
    note=spec["note"] if not (jeu=="seismes" and seismes_url) else "USGS seismes (fenetre/region ciblee)"
    path=download(url, dest)
    if path is None: return None, note
    df=pd.read_csv(path); c=spec["col"]
    if spec["unit"]=="days":   # colonne datetime -> jours
        t=pd.to_datetime(df[c], errors="coerce").dropna()
        ev=np.sort((t-t.min()).dt.total_seconds().values/86400.0)
    else:                       # deja numerique (annees fractionnaires)
        v=pd.to_numeric(df[c], errors="coerce").dropna().values.astype(float); ev=np.sort(v)
    return ev-ev.min(), spec["note"]

# ---------------- estimateurs renouvellement (inter-arrivees) --------------
FAMS={"Weibull":stats.weibull_min, "Gamma":stats.gamma, "Lognormal":stats.lognorm}
def fam_mle(dist,d):
    c,loc,sc=dist.fit(d, floc=0); return (c,sc)
def _kde(d,grid,h):
    z=(grid[None,:]-d[:,None])/h; g=(np.exp(-0.5*z*z)/(h*np.sqrt(2*np.pi))).mean(0)
    zr=(grid[None,:]+d[:,None])/h; return g+(np.exp(-0.5*zr*zr)/(h*np.sqrt(2*np.pi))).mean(0)
def fam_mhde(dist,d,th0):
    if len(d)<8: return th0
    scale=min(np.std(d),(np.subtract(*np.percentile(d,[75,25]))/1.349) or np.std(d))
    h=max(scale*len(d)**(-1/3.0),1e-3)
    xmax=np.quantile(d,0.999)*1.3+1e-6; grid=np.linspace(1e-6,xmax,400); dx=grid[1]-grid[0]
    sg=np.sqrt(np.clip(_kde(d,grid,h),0,None))
    def obj(lp):
        th=np.exp(lp); f=np.clip(dist.pdf(grid,th[0],scale=th[1]),1e-12,None)
        return np.sum((sg-np.sqrt(f))**2)*dx
    return tuple(np.exp(optimize.minimize(obj,np.log(np.clip(th0,1e-6,None)),method="Nelder-Mead",
                options=dict(maxiter=400,xatol=1e-4,fatol=1e-6)).x))
def fam_dpd(dist,d,th0,a=0.5):
    xmax=np.quantile(d,0.999)*1.3+1e-6; grid=np.linspace(1e-6,xmax,400); dx=grid[1]-grid[0]
    def obj(lp):
        th=np.exp(lp); fg=np.clip(dist.pdf(grid,th[0],scale=th[1]),1e-300,None)
        fe=np.clip(dist.pdf(d,th[0],scale=th[1]),1e-300,None)
        return np.sum(fg**(1+a))*dx-(1+1/a)*np.mean(fe**a)
    return tuple(np.exp(optimize.minimize(obj,np.log(np.clip(th0,1e-6,None)),method="Nelder-Mead",
                options=dict(maxiter=400,xatol=1e-4,fatol=1e-6)).x))

def analyse_renouvellement(ev, resdir):
    x=np.diff(ev); x=x[x>0]
    print(f"\n=== (1) RENOUVELLEMENT : {len(x)} inter-arrivees, moyenne={x.mean():.4g} ===", flush=True)
    print(f"{'famille':10s} {'estimateur':16s} {'p1':>9s} {'p2':>9s} {'GOF p (KS)':>11s} {'AIC':>9s} {'sens.1out':>10s}")
    rows=[]; best=None
    xo=np.append(x, x.max()*6.0)
    for fname,dist in FAMS.items():
        mle=fam_mle(dist,x); mh=fam_mhde(dist,x,mle); dp=fam_dpd(dist,x,mle)
        ll=np.sum(np.log(np.clip(dist.pdf(x,mle[0],scale=mle[1]),1e-300,None))); aic=4-2*ll
        for ename,th in [("MLE",mle),("MHDE (sous-lisse)",mh),("DPD(0.5)",dp)]:
            p=float(stats.kstest(x,dist.name,args=(th[0],0,th[1])).pvalue)
            if ename=="MLE": th2=fam_mle(dist,xo)
            elif ename.startswith("MHDE"): th2=fam_mhde(dist,xo,fam_mle(dist,xo))
            else: th2=fam_dpd(dist,xo,fam_mle(dist,xo))
            sens=float(np.linalg.norm(np.array(th2)-np.array(th)))
            aic_str=f"{aic:9.1f}" if ename=="MLE" else " "*9
            print(f"{fname:10s} {ename:16s} {th[0]:9.3f} {th[1]:9.3f} {p:11.3f} "
                  f"{aic_str} {sens:10.3f}")
            rows.append((fname,ename,round(th[0],4),round(th[1],4),round(p,4),
                         round(aic,2) if ename=="MLE" else "",round(sens,4)))
        if best is None or aic<best[1]: best=(fname,aic)
    print(f"  -> meilleure famille par AIC : {best[0]}")
    _csv(resdir,"renouvellement.csv",
         ["famille","estimateur","p1","p2","GOF_p_KS","AIC","sens_1outlier"],rows)
    return x, best[0]

# ---------------- Hawkes (auto-excitation), beta libre ---------------------
def hawkes_A(ev, beta):
    A=np.zeros(len(ev))
    for i in range(1,len(ev)): A[i]=np.exp(-beta*(ev[i]-ev[i-1]))*(1.0+A[i-1])
    return A
def hawkes_nll(th, ev, T):
    mu,a,beta=th
    if mu<=0 or a<=0 or a>=1 or beta<=0: return 1e12
    A=hawkes_A(ev,beta); li=mu+a*beta*A
    if np.any(li<=0): return 1e12
    return mu*T + a*np.sum(1-np.exp(-beta*(T-ev))) - np.sum(np.log(li))
def hawkes_grid_int(ev,T,th,m=1500):
    mu,a,beta=th; grid=np.linspace(0,T,m); dg=grid[1]-grid[0]; K=np.zeros(m); S=0.0; idx=0
    for k in range(m):
        if k>0: S*=np.exp(-beta*dg)
        while idx<len(ev) and ev[idx]<=grid[k]: S+=np.exp(-beta*(grid[k]-ev[idx])); idx+=1
        K[k]=S
    return np.clip(mu+a*beta*K,1e-300,None), dg
def hawkes_dpd_obj(th, ev, T, alpha):
    mu,a,beta=th
    if mu<=0 or a<=0 or a>=1 or beta<=0: return 1e12
    lam_grid,dg=hawkes_grid_int(ev,T,th); A=hawkes_A(ev,beta); le=np.clip(mu+a*beta*A,1e-300,None)
    return np.sum(lam_grid**(1+alpha))*dg - (1+1/alpha)*np.sum(le**alpha)
def hawkes_gof(ev, th):
    mu,a,beta=th; k=np.arange(len(ev)); A=hawkes_A(ev,beta)
    Lam=mu*ev + a*(k - A); dL=np.diff(Lam); dL=dL[dL>=0]
    return float(stats.kstest(dL,'expon').pvalue) if len(dL)>=10 else np.nan

def analyse_hawkes(ev, resdir):
    T=ev.max()*1.0001; n=len(ev)
    print(f"\n=== (2) HAWKES exponentiel : {n} evenements sur [0,{T:.4g}] ===", flush=True)
    th0=[n/T*0.5, 0.3, 5.0/np.median(np.diff(ev)+1e-9)]
    mle=optimize.minimize(hawkes_nll,th0,args=(ev,T),method="Nelder-Mead",
                          options=dict(maxiter=1000,maxfev=1500,xatol=1e-4,fatol=1e-4)).x
    dpd=optimize.minimize(hawkes_dpd_obj,mle,args=(ev,T,0.5),method="Nelder-Mead",
                          options=dict(maxiter=800,maxfev=1200,xatol=1e-4,fatol=1e-4)).x
    rows=[]
    print(f"{'estimateur':12s} {'mu':>9s} {'a (branch.)':>12s} {'beta':>9s} {'GOF p':>8s}")
    for name,th in [("MLE",mle),("DPD(0.5)",dpd)]:
        p=hawkes_gof(ev,th)
        print(f"{name:12s} {th[0]:9.4f} {th[1]:12.3f} {th[2]:9.4f} {p:8.3f}")
        rows.append((name,round(th[0],5),round(th[1],4),round(th[2],5),round(p,4)))
    a_hat=mle[1]
    print(f"  -> ratio de branchement (noyau exp.) a={a_hat:.3f} "
          f"({'auto-excitation nette' if a_hat>0.2 else 'faible auto-excitation'})")
    _csv(resdir,"hawkes.csv",["estimateur","mu","a_branchement","beta","GOF_p"],rows)
    return mle, hawkes_gof(ev,mle)

# ---------------- Hawkes a noyau Omori/ETAS (loi de puissance) -------------
# Parametrisation bornee (evite la degenerescence sur donnees ~exponentielles) :
#   mu=exp(.)  kappa=exp(.)  c=1e-4+CMAX*expit(.)  p=1+2*expit(.) in (1,3)
from scipy.special import expit
def etas_unpack(par, cmax):
    return (np.exp(par[0]), np.exp(par[1]), 1e-4+cmax*expit(par[2]), 1.0+2.0*expit(par[3]))
def etas_lambda_events(ev, mu, kappa, c, p):
    lam=np.full(len(ev), mu)
    for i in range(1,len(ev)): lam[i]+=kappa*np.sum((ev[i]-ev[:i]+c)**(-p))
    return lam
def etas_compensator(ev, T, mu, kappa, c, p):
    return mu*T + kappa/(1-p)*np.sum((T-ev+c)**(1-p)-c**(1-p))
def etas_nll(par, ev, T, cmax):
    mu,kappa,c,p=etas_unpack(par,cmax)
    lam=etas_lambda_events(ev,mu,kappa,c,p)
    if np.any(lam<=0) or not np.isfinite(lam).all(): return 1e12
    val=etas_compensator(ev,T,mu,kappa,c,p)-np.sum(np.log(lam))
    return val if np.isfinite(val) else 1e12
def etas_Lam(ev, mu, kappa, c, p):
    Lam=mu*ev.astype(float).copy()
    for i in range(1,len(ev)): Lam[i]+=kappa/(1-p)*np.sum((ev[i]-ev[:i]+c)**(1-p)-c**(1-p))
    return Lam
def etas_gof(ev, th):
    mu,kappa,c,p=th; dL=np.diff(etas_Lam(ev,mu,kappa,c,p)); dL=dL[dL>=0]
    return float(stats.kstest(dL,'expon').pvalue) if len(dL)>=10 else np.nan
def etas_branch(th):
    mu,kappa,c,p=th; return kappa*c**(1-p)/(p-1) if p>1 else np.inf

def analyse_etas(ev, resdir):
    T=ev.max()*1.0001; n=len(ev); med=np.median(np.diff(ev))+1e-9; cmax=2.0*med
    print(f"\n=== (3) HAWKES a noyau Omori/ETAS (loi de puissance) : {n} evenements ===", flush=True)
    if n>2500:
        print(f"    [patience] ajustement ETAS en O(N^2) sur {n} evenements : ~1-2 min...", flush=True)
    starts=[(1.1,0.1),(1.4,0.3)]                    # 2 starts (vraisemblance multimodale)
    best=None
    for p0,cf in starts:
        par0=[np.log(n/T*0.5), np.log(0.3),
              np.log(max(cf*med-1e-4,1e-6)/max(cmax-(cf*med),1e-6)),
              np.log((p0-1.0)/(3.0-p0))]
        r=optimize.minimize(etas_nll,par0,args=(ev,T,cmax),method="Nelder-Mead",
                            options=dict(maxiter=400,maxfev=600,xatol=1e-4,fatol=1e-4))
        if best is None or r.fun<best.fun: best=r
    th=etas_unpack(best.x,cmax); mu,kappa,c,p=th; eta=etas_branch(th); gp=etas_gof(ev,th)
    print(f"{'estimateur':12s} {'mu':>8s} {'kappa':>8s} {'c':>8s} {'p (Omori)':>10s} {'branch.':>8s} {'GOF p':>8s}")
    print(f"{'MLE':12s} {mu:8.4f} {kappa:8.4f} {c:8.5f} {p:10.3f} {eta:8.3f} {gp:8.3f}")
    print(f"  -> p (decroissance Omori) = {p:.2f} ; ratio de branchement eta = {eta:.3f}")
    _csv(resdir,"etas.csv",["estimateur","mu","kappa","c","p_omori","branchement","GOF_p"],
         [("MLE",round(mu,5),round(kappa,5),round(c,6),round(p,4),round(eta,4),round(gp,4))])
    return th, gp

# ---------------- sortie + figure ------------------------------------------
def _csv(resdir,name,header,rows):
    os.makedirs(resdir,exist_ok=True)
    with open(os.path.join(resdir,name),"w",newline="") as f:
        w=csv.writer(f); w.writerow(header)
        for r in rows: w.writerow(list(r))

def figure(figdir, ev, x, best_fam, hawkes_th, etas_th, hgof, egof, note):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e: print(f"[info] figure ignoree ({e})"); return None
    os.makedirs(figdir,exist_ok=True); dist=FAMS[best_fam]
    fig,axs=plt.subplots(1,3,figsize=(16,4.6))
    # comptage cumule + intensite Hawkes ajustee
    axs[0].step(ev,np.arange(1,len(ev)+1),where='post',color='0.4',lw=1,label='N(t) observe')
    mu,a,beta=hawkes_th; T=ev.max()
    lam,dg=hawkes_grid_int(ev,T,hawkes_th,m=1500); grid=np.linspace(0,T,1500)
    ax2=axs[0].twinx(); ax2.plot(grid,lam,color='tab:red',lw=0.8,alpha=0.7,label='intensite Hawkes')
    axs[0].set_xlabel("temps"); axs[0].set_ylabel("comptage cumule N(t)")
    ax2.set_ylabel("intensite ajustee",color='tab:red'); axs[0].set_title("Comptage + intensite Hawkes",fontsize=9)
    # inter-arrivees + densite ajustee (meilleure famille, MLE)
    mle=fam_mle(dist,x); xs=np.linspace(1e-6,np.quantile(x,0.99),200)
    axs[1].hist(x,bins=40,density=True,alpha=0.6,color='tab:blue')
    axs[1].plot(xs,dist.pdf(xs,mle[0],scale=mle[1]),'r-',lw=1.5,label=f'{best_fam} (MLE)')
    axs[1].set_xlabel("inter-arrivee"); axs[1].set_ylabel("densite")
    axs[1].set_title(f"(1) Inter-arrivees + {best_fam}",fontsize=9); axs[1].legend(fontsize=8)
    # GOF time-rescaling : exponentiel vs Omori/ETAS
    def qq(dL):
        dq=np.sort(dL[dL>=0]); q=stats.expon.ppf((np.arange(1,len(dq)+1)-0.5)/len(dq)); return q,dq
    k=np.arange(len(ev)); A=hawkes_A(ev,beta); Lh=mu*ev+a*(k-A)
    qh,dh=qq(np.diff(Lh))
    Le=etas_Lam(ev,*etas_th); qe,de=qq(np.diff(Le))
    axs[2].plot(qh,dh,'o',ms=3,color='0.6',alpha=0.5,label=f'exponentiel (p={hgof:.3f})')
    axs[2].plot(qe,de,'o',ms=3,color='tab:green',alpha=0.6,label=f'Omori/ETAS (p={egof:.3f})')
    lim=[0,max(qh.max(),dh.max(),qe.max(),de.max())]; axs[2].plot(lim,lim,'r-',lw=1)
    axs[2].set_xlabel("quantiles Exp(1)"); axs[2].set_ylabel("increments re-echelonnes")
    axs[2].set_title("(2)+(3) GOF Hawkes : exp. vs ETAS",fontsize=9); axs[2].legend(fontsize=8); axs[2].grid(alpha=.3)
    fig.suptitle(note,fontsize=10); fig.tight_layout()
    p=os.path.join(figdir,"reelles_benchmark.png"); fig.savefig(p,dpi=130); plt.close(fig)
    return p

def demo_events(rng):
    """Repli sans reseau : epoques Hawkes synthetiques (pour verifier le pipeline)."""
    mu,a,beta=2.0,0.5,1.0; Tdemo=200.0; t=0.0; ev=[]; last=0.0; S=0.0
    while True:
        lb=mu+a*beta*S; t+=rng.exponential(1.0/lb)
        if t>=Tdemo: break
        S*=np.exp(-beta*(t-last)); last=t
        if rng.uniform()<=(mu+a*beta*S)/lb: ev.append(t); S+=1.0
    return np.array(ev)-0.0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--jeu",choices=list(CATALOG),default="seismes")
    ap.add_argument("--csv"); ap.add_argument("--col"); ap.add_argument("--mode",choices=["times","gaps"],default="times")
    ap.add_argument("--start"); ap.add_argument("--end"); ap.add_argument("--minmag",type=float)
    ap.add_argument("--bbox", help="minlat,maxlat,minlon,maxlon : isole une sequence regionale")
    ap.add_argument("--list",action="store_true")
    a=ap.parse_args()
    if a.list:
        for k,v in CATALOG.items(): print(f"  {k:10s} {v['note']}\n             {v['url']}")
        print("\n  Exemple sequence regionale (repliques, 1 noyau Hawkes) :")
        print("    --jeu seismes --start 2023-02-06 --end 2023-03-31 --minmag 4.0 \\")
        print("        --bbox 35,39,35,40   (region Turquie-Syrie 2023)")
        return
    base=os.path.join(script_dir(),OUTPUT_DIRNAME)
    resdir=os.path.join(base,RESULTS_SUBDIR); figdir=os.path.join(base,FIGURE_SUBDIR); datadir=os.path.join(base,DATA_SUBDIR)
    for d in (resdir,figdir,datadir): os.makedirs(d,exist_ok=True)
    print(f"Sortie -> {base}")
    # URL seismes personnalisee si l'utilisateur cible une fenetre/region
    su=None; suf=""
    if a.jeu=="seismes" and (a.start or a.end or a.minmag is not None or a.bbox):
        bb=[float(x) for x in a.bbox.split(",")] if a.bbox else None
        su=usgs_url(a.start or "2024-01-01", a.end or "2024-12-31",
                    a.minmag if a.minmag is not None else 4.5, bb)
        suf="_cible"
    ev,note=load_event_times(a.jeu,a.csv,a.col,a.mode,datadir,seismes_url=su,suffix=suf)
    if ev is None or len(ev)<30:
        print("[repli] pas de reseau ou trop peu d'evenements -> DEMO Hawkes synthetique")
        ev=demo_events(np.random.default_rng(0)); note="DEMO synthetique (pas de reseau)"
    print(f"\nJEU : {note}  ->  {len(ev)} epoques observees, etendue [0,{ev.max():.4g}]")
    print("ATTENTION : donnees reelles, pas de verite-terrain => efficacite NON mesuree.")
    x,best=analyse_renouvellement(ev,resdir)
    hth,hgof=analyse_hawkes(ev,resdir)
    eth,egof=analyse_etas(ev,resdir)
    print(f"\n=== COMPARAISON noyaux Hawkes (GOF time-rescaling) ===")
    print(f"  exponentiel : GOF p = {hgof:.4f}")
    print(f"  Omori/ETAS  : GOF p = {egof:.4f}")
    verdict=("ETAS ajuste mieux (loi de puissance)" if egof>max(hgof,0.05)
             else "aucun noyau n'ajuste (p<0.05) -> modele encore trop simple"
             if max(hgof,egof)<0.05 else "exponentiel suffit")
    print(f"  -> {verdict}")
    p=figure(figdir,ev,x,best,hth,eth,hgof,egof,note)
    print(f"\nEcrit : {resdir}/  (csv)")
    if p: print(f"        {p}")

if __name__=="__main__":
    main()
