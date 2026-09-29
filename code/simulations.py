import warnings; warnings.filterwarnings("ignore")
import os
import numpy as np
from scipy import optimize, stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else ".", "figures")
os.makedirs(OUTDIR, exist_ok=True)

def simulate_ipp(th, T, rng):
    hi = th[0] + abs(th[1]); lmax = np.exp(hi)
    n = rng.poisson(lmax*T); cand = np.sort(rng.uniform(0, T, n))
    keep = rng.uniform(0, 1, n) < np.exp(th[0] + th[1]*np.sin(2*np.pi*cand/(T/3.0)))/lmax
    return cand[keep]

def lam_ipp(t, th, T): return np.exp(th[0] + th[1]*np.sin(2*np.pi*t/(T/3.0)))

def fit_ipp(ev, T, alpha=None):
    grid = np.linspace(0, T, 600); dg = grid[1]-grid[0]; th0 = np.array([np.log(max(len(ev),1)/T), 0.0])
    def nll(th): return np.sum(lam_ipp(grid, th, T))*dg - np.sum(np.log(np.clip(lam_ipp(ev, th, T),1e-300,None)))
    mle = optimize.minimize(nll, th0, method="Nelder-Mead", options=dict(maxiter=600,xatol=1e-4,fatol=1e-6)).x
    if alpha is None: return mle
    def obj(th):
        l = np.clip(lam_ipp(grid,th,T),1e-300,None); le = np.clip(lam_ipp(ev,th,T),1e-300,None)
        return np.sum(l**(1+alpha))*dg - (1+1/alpha)*np.sum(le**alpha)
    return optimize.minimize(obj, mle, method="Nelder-Mead", options=dict(maxiter=600,xatol=1e-4,fatol=1e-6)).x

def contaminate_ipp(ev, T, eps, rng):
    m = int(eps*max(len(ev),1))
    if m <= 0: return ev
    P = T/3.0; troughs = [0.75*P+k*P for k in range(3) if 0.75*P+k*P < T]
    spur = np.clip(rng.choice(troughs, m)+rng.normal(0,P*0.03,m), 0, T)
    return np.sort(np.concatenate([ev, spur]))

def simulate_hawkes(T, rng, mu, a, beta):
    t=0.0; ev=[]; last=0.0; S=0.0
    while True:
        lb=mu+a*beta*S; t+=rng.exponential(1.0/lb)
        if t>=T: break
        S*=np.exp(-beta*(t-last)); last=t
        if rng.uniform()<=(mu+a*beta*S)/lb: ev.append(t); S+=1.0
    return np.array(ev)

def hawkes_A(ev, beta):
    A=np.zeros(len(ev))
    for i in range(1,len(ev)): A[i]=np.exp(-beta*(ev[i]-ev[i-1]))*(1.0+A[i-1])
    return A

def hawkes_nll(th, ev, T, beta):
    mu,a=th
    if mu<=0 or a<=0 or a>=1: return 1e12
    A=hawkes_A(ev,beta); li=mu+a*beta*A
    if np.any(li<=0): return 1e12
    return mu*T+a*np.sum(1-np.exp(-beta*(T-ev)))-np.sum(np.log(li))

def hawkes_dpd(th, ev, T, beta, alpha, grid):
    mu,a=th
    if mu<=0 or a<=0 or a>=1: return 1e12
    dg=grid[1]-grid[0]; K=np.zeros(len(grid)); S=0.0; idx=0
    for k in range(len(grid)):
        if k>0: S*=np.exp(-beta*dg)
        while idx<len(ev) and ev[idx]<=grid[k]: S+=np.exp(-beta*(grid[k]-ev[idx])); idx+=1
        K[k]=S
    lg=np.clip(mu+a*beta*K,1e-300,None); A=hawkes_A(ev,beta); le=np.clip(mu+a*beta*A,1e-300,None)
    return np.sum(lg**(1+alpha))*dg-(1+1/alpha)*np.sum(le**alpha)

def mle_gamma(d): k,loc,sc=stats.gamma.fit(d,floc=0); return np.array([k,sc])
def kde(d,grid,h):
    z=(grid[None,:]-d[:,None])/h; g=(np.exp(-0.5*z*z)/(h*np.sqrt(2*np.pi))).mean(0)
    zr=(grid[None,:]+d[:,None])/h; return g+(np.exp(-0.5*zr*zr)/(h*np.sqrt(2*np.pi))).mean(0)
def bw(d,rule):
    sc=min(np.std(d),(np.subtract(*np.percentile(d,[75,25]))/1.349) or np.std(d)); n=len(d)
    return max(sc*n**(-1/3.0) if rule=="us" else 0.9*sc*n**(-1/5.0), 1e-3)
def mhde_gamma(d,th0,rule):
    if len(d)<8: return th0
    xmax=np.quantile(d,0.999)*1.3+1e-6; grid=np.linspace(1e-6,xmax,400); dx=grid[1]-grid[0]
    sg=np.sqrt(np.clip(kde(d,grid,bw(d,rule)),0,None))
    def obj(lp):
        th=np.exp(lp); f=np.clip(stats.gamma.pdf(grid,th[0],scale=th[1]),1e-12,None)
        return np.sum((sg-np.sqrt(f))**2)*dx
    return np.exp(optimize.minimize(obj,np.log(np.clip(th0,1e-6,None)),method="Nelder-Mead",options=dict(maxiter=400,xatol=1e-4,fatol=1e-6)).x)
def dpd_gamma(d,th0,a=0.5):
    xmax=np.quantile(d,0.999)*1.3+1e-6; grid=np.linspace(1e-6,xmax,400); dx=grid[1]-grid[0]
    def obj(lp):
        th=np.exp(lp); fg=np.clip(stats.gamma.pdf(grid,th[0],scale=th[1]),1e-300,None)
        fe=np.clip(stats.gamma.pdf(d,th[0],scale=th[1]),1e-300,None)
        return np.sum(fg**(1+a))*dx-(1+1/a)*np.mean(fe**a)
    return np.exp(optimize.minimize(obj,np.log(np.clip(th0,1e-6,None)),method="Nelder-Mead",options=dict(maxiter=400,xatol=1e-4,fatol=1e-6)).x)


# robustesse_renouvellement.png
def fig_robustesse_renouvellement(reps=150):
    true=np.array([2.0,1.5]); n=300
    def run(eps,seed):
        rng=np.random.default_rng(seed); sq={"MLE":[],"MHDE":[],"DPD(0.5)":[]}
        for _ in range(reps):
            d=stats.gamma.rvs(true[0],scale=true[1],size=n,random_state=rng)
            if eps>0:
                k=int(eps*n); idx=rng.choice(n,k,replace=False); d=d.copy(); d[idx]*=6.0
            m=mle_gamma(d)
            sq["MLE"].append(np.sum((m-true)**2)); sq["MHDE"].append(np.sum((mhde_gamma(d,m,"us")-true)**2)); sq["DPD(0.5)"].append(np.sum((dpd_gamma(d,m)-true)**2))
        return {k:float(np.mean(v)) for k,v in sq.items()}
    m0=run(0.0,3); mc=run(0.05,4); ks=["MLE","MHDE","DPD(0.5)"]; cols=["0.6","tab:green","tab:blue"]
    fig,axs=plt.subplots(1,2,figsize=(11,4.6))
    axs[0].bar(ks,[m0[k] for k in ks],color=cols); axs[0].set_title("Renouvellement Gamma — au modele (MSE)"); axs[0].set_ylabel("MSE")
    axs[1].bar(ks,[mc[k] for k in ks],color=cols); axs[1].set_title("Renouvellement Gamma — 5% aberrants (MSE)"); axs[1].set_ylabel("MSE")
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR,"robustesse_renouvellement.png"),dpi=130); plt.close(fig)


# nonrenouvellement_benchmark.png
def fig_nonrenouvellement(reps=150):
    th=np.array([np.log(8.0),1.0]); T0=60; eps=0.10
    def mse(T,e,alpha,seed):
        rng=np.random.default_rng(seed); sq={"MLE":[],"DPD(0.1)":[],"DPD(0.5)":[]}
        for _ in range(reps):
            ev=simulate_ipp(th,T,rng)
            if len(ev)<8: continue
            ec=contaminate_ipp(ev,T,e,rng) if e>0 else ev
            sq["MLE"].append(np.sum((fit_ipp(ec,T)-th)**2))
            sq["DPD(0.1)"].append(np.sum((fit_ipp(ec,T,0.1)-th)**2))
            sq["DPD(0.5)"].append(np.sum((fit_ipp(ec,T,0.5)-th)**2))
        return {k:float(np.mean(v)) for k,v in sq.items()}
    m0=mse(T0,0.0,0.5,11); mc=mse(T0,eps,0.5,22)
    Ts=[T0,2*T0,4*T0,8*T0]; c1=[]; c5=[]
    for T in Ts:
        mt=mse(T,0.0,0.5,33); c1.append(mt["MLE"]/mt["DPD(0.1)"]); c5.append(mt["MLE"]/mt["DPD(0.5)"])
    fig,axs=plt.subplots(1,3,figsize=(15,4.5)); ks=["MLE","DPD(0.1)","DPD(0.5)"]; cols=["0.6","tab:green","tab:blue"]
    axs[0].bar(ks,[m0["MLE"]/m0[k] for k in ks],color=cols); axs[0].axhline(1,ls='--',color='0.5')
    axs[0].set_title("(A1) Efficacite au modele"); axs[0].set_ylabel("efficacite")
    axs[1].bar(ks,[mc[k] for k in ks],color=cols); axs[1].set_title("(A2) Robustesse (parasites)"); axs[1].set_ylabel("MSE")
    axs[2].plot(Ts,c1,'-o',label="DPD(0.1)",color="tab:green"); axs[2].plot(Ts,c5,'-o',label="DPD(0.5)",color="tab:blue")
    axs[2].axhline(1,ls='--',color='0.5'); axs[2].set_title("(A3) Efficacite vs T"); axs[2].set_xlabel("T"); axs[2].legend(); axs[2].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR,"nonrenouvellement_benchmark.png"),dpi=130); plt.close(fig)


# rupture_benchmark.png
def fig_rupture(reps=300):
    tau0=0.4; ra=2.0; rb=5.0
    def estim_tau(ev):
        if len(ev)<4: return 0.5
        cand=ev; best=-1e18; bt=0.5
        for j in range(1,len(ev)):
            t=ev[j]; nl=j; nr=len(ev)-j
            if nl<1 or nr<1 or t<=0 or t>=1: continue
            ll=nl*np.log(nl/t)+nr*np.log(nr/(1-t))
            if ll>best: best=ll; bt=t
        return bt
    ns=[50,100,200,400,800]; mse=[]; errs_big=None
    for n in ns:
        rng=np.random.default_rng(100+n); e=[]
        for _ in range(reps):
            na=rng.poisson(n*ra*tau0); nb=rng.poisson(n*rb*(1-tau0))
            ev=np.sort(np.concatenate([rng.uniform(0,tau0,na),rng.uniform(tau0,1,nb)]))
            e.append((estim_tau(ev)-tau0))
        e=np.array(e); mse.append(np.mean(e**2))
        if n==800: errs_big=n*e
    slope=np.polyfit(np.log(ns[1:]),np.log(mse[1:]),1)[0]
    fig,axs=plt.subplots(1,2,figsize=(11,4.6))
    axs[0].plot(np.log(ns),np.log(mse),'-o',color="tab:blue")
    axs[0].set_xlabel("log n"); axs[0].set_ylabel("log MSE(tau)")
    axs[0].set_title(f"(B1) Rate de rupture : pente={slope:.2f} (rate n=> -2)"); axs[0].grid(alpha=.3)
    axs[1].hist(errs_big,bins=40,density=True,color="tab:purple",alpha=0.7)
    axs[1].set_xlabel("n (tau_hat - tau0)"); axs[1].set_ylabel("densite")
    axs[1].set_title(f"(B2) Loi limite non gaussienne (kurtosis={stats.kurtosis(errs_big):.1f})"); axs[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR,"rupture_benchmark.png"),dpi=130); plt.close(fig)


# hawkes_benchmark.png
def fig_hawkes(reps=200):
    mu0,a0,beta=1.0,0.5,1.0; T=100.0; th0=np.array([mu0,a0])
    grid=np.linspace(0,T,700)
    rng=np.random.default_rng(7); sq_mle=[]; sq_dpd=[]; z=[]; cover=0
    for _ in range(reps):
        ev=simulate_hawkes(T,rng,mu0,a0,beta)
        if len(ev)<20: continue
        mle=optimize.minimize(hawkes_nll,th0,args=(ev,T,beta),method="Nelder-Mead",options=dict(maxiter=400,xatol=1e-4,fatol=1e-6)).x
        dpd=optimize.minimize(hawkes_dpd,mle,args=(ev,T,beta,0.5,grid),method="Nelder-Mead",options=dict(maxiter=300,xatol=1e-4,fatol=1e-6)).x
        sq_mle.append(np.sum((mle-th0)**2)); sq_dpd.append(np.sum((dpd-th0)**2))
        h=1e-4; H=np.zeros((2,2)); f0=hawkes_nll(mle,ev,T,beta)
        for i in range(2):
            for j in range(2):
                ei=np.zeros(2); ei[i]=h; ej=np.zeros(2); ej[j]=h
                H[i,j]=(hawkes_nll(mle+ei+ej,ev,T,beta)-hawkes_nll(mle+ei,ev,T,beta)-hawkes_nll(mle+ej,ev,T,beta)+f0)/(h*h)
        try: V=np.linalg.inv(H); se=np.sqrt(abs(V[0,0]))
        except Exception: continue
        zz=(mle[0]-mu0)/se; z.append(zz)
        if abs(zz)<=1.96: cover+=1
    z=np.array(z); cov=cover/max(len(z),1)
    rng2=np.random.default_rng(8); g_ok=[]; g_bad=[]
    for _ in range(80):
        ev=simulate_hawkes(T,rng2,mu0,a0,beta)
        if len(ev)<20: continue
        mle=optimize.minimize(hawkes_nll,th0,args=(ev,T,beta),method="Nelder-Mead",options=dict(maxiter=400,xatol=1e-4,fatol=1e-6)).x
        k=np.arange(len(ev)); A=hawkes_A(ev,beta); Lam=mle[0]*ev+mle[1]*(k-A); dL=np.diff(Lam); dL=dL[dL>=0]
        g_ok.append(stats.kstest(dL,'expon').pvalue)
        lam_p=len(ev)/T; dLp=lam_p*np.diff(ev)
        g_bad.append(stats.kstest(dLp,'expon').pvalue)
    sq_mle=np.array(sq_mle); sq_dpd=np.array(sq_dpd)
    fig,axs=plt.subplots(2,2,figsize=(12,9))
    axs[0,0].bar(["MLE","DPD(0.5)"],[np.mean(sq_mle),np.mean(sq_dpd)],color=["0.6","tab:blue"])
    axs[0,0].set_title("(C1) Efficacite au modele (MSE)")
    rng3=np.random.default_rng(9); rm=[]; rd=[]
    for _ in range(120):
        ev=simulate_hawkes(T,rng3,mu0,a0,beta)
        if len(ev)<20: continue
        par=np.sort(rng3.uniform(0,T,int(0.15*len(ev)))); ec=np.sort(np.concatenate([ev,par]))
        mle=optimize.minimize(hawkes_nll,th0,args=(ec,T,beta),method="Nelder-Mead",options=dict(maxiter=400,xatol=1e-4,fatol=1e-6)).x
        dpd=optimize.minimize(hawkes_dpd,mle,args=(ec,T,beta,0.5,grid),method="Nelder-Mead",options=dict(maxiter=300,xatol=1e-4,fatol=1e-6)).x
        rm.append(np.sum((mle-th0)**2)); rd.append(np.sum((dpd-th0)**2))
    axs[0,1].bar(["MLE","DPD(0.5)"],[np.mean(rm),np.mean(rd)],color=["0.6","tab:blue"])
    axs[0,1].set_title("(C2) Robustesse (15% parasites, MSE)")
    qq=np.sort(z); qn=stats.norm.ppf((np.arange(1,len(qq)+1)-0.5)/len(qq))
    axs[1,0].plot(qn,qq,'o',ms=3,color="tab:purple",alpha=0.6); lim=[qn.min(),qn.max()]; axs[1,0].plot(lim,lim,'r-')
    axs[1,0].set_title(f"(C3) Normalite + sandwich (couverture IC95%={cov:.0%})"); axs[1,0].set_xlabel("quantiles normaux"); axs[1,0].grid(alpha=.3)
    axs[1,1].hist(g_ok,bins=12,range=(0,1),alpha=0.6,color="tab:green",label="Hawkes (vrai)")
    axs[1,1].hist(g_bad,bins=12,range=(0,1),alpha=0.6,color="tab:red",label="Poisson (faux)")
    axs[1,1].set_title("(C4) GOF time-rescaling (p-values)"); axs[1,1].set_xlabel("p-value"); axs[1,1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR,"hawkes_benchmark.png"),dpi=130); plt.close(fig)


# conjecture1_benchmark.png
def fig_conjecture1(reps=120):
    true=np.array([2.0,1.5]); ns=[100,200,400,800,1600]
    cur={"MHDE (sous-lisse)":[],"MHDE (Silverman)":[],"DPD(0.5)":[]}
    for n in ns:
        rng=np.random.default_rng(1+n); sm=[];sh=[];ss=[];sd=[]
        for _ in range(reps):
            d=stats.gamma.rvs(true[0],scale=true[1],size=n,random_state=rng); m=mle_gamma(d)
            sm.append(np.sum((m-true)**2)); sh.append(np.sum((mhde_gamma(d,m,"us")-true)**2))
            ss.append(np.sum((mhde_gamma(d,m,"silverman")-true)**2)); sd.append(np.sum((dpd_gamma(d,m)-true)**2))
        em=np.mean(sm)
        cur["MHDE (sous-lisse)"].append(em/np.mean(sh)); cur["MHDE (Silverman)"].append(em/np.mean(ss)); cur["DPD(0.5)"].append(em/np.mean(sd))
    rng=np.random.default_rng(7); rob={"MLE":[],"MHDE (sous-lisse)":[],"DPD(0.5)":[]}
    for _ in range(reps):
        d=stats.gamma.rvs(true[0],scale=true[1],size=400,random_state=rng)
        idx=rng.choice(400,20,replace=False); d=d.copy(); d[idx]*=6.0; m=mle_gamma(d)
        rob["MLE"].append(np.sum((m-true)**2)); rob["MHDE (sous-lisse)"].append(np.sum((mhde_gamma(d,m,"us")-true)**2)); rob["DPD(0.5)"].append(np.sum((dpd_gamma(d,m)-true)**2))
    rob={k:float(np.mean(v)) for k,v in rob.items()}
    fig,axs=plt.subplots(1,2,figsize=(13,5)); cm={"MHDE (sous-lisse)":"tab:green","MHDE (Silverman)":"tab:orange","DPD(0.5)":"tab:blue"}
    for k in cur: axs[0].plot(ns,cur[k],'-o',label=k,color=cm[k])
    axs[0].axhline(1,ls='--',color='0.6'); axs[0].set_xscale('log'); axs[0].set_xlabel("n"); axs[0].set_ylabel("efficacite")
    axs[0].set_title("(D1) Efficacite vs n"); axs[0].legend(fontsize=8); axs[0].grid(alpha=.3)
    ks=list(rob); axs[1].bar(ks,[rob[k] for k in ks],color=['0.6','tab:green','tab:blue'])
    axs[1].set_title("(D2) Robustesse (5% aberrants)"); axs[1].set_ylabel("MSE")
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR,"conjecture1_benchmark.png"),dpi=130); plt.close(fig)


# conditions_benchmark.png
def fig_conditions(Tmax=1500, reps=6, alpha=0.0, eps=0.1):
    MU0,A0,BETA=1.0,0.5,1.0; m=max(2000,Tmax*2); hor=np.linspace(Tmax*0.05,Tmax,18)
    SigA=np.zeros((len(hor),3)); JA=np.zeros((len(hor),3)); L0=np.zeros(len(hor)); L5=np.zeros(len(hor)); nr=0
    for r in range(reps):
        rng=np.random.default_rng(11+r); ev=simulate_hawkes(Tmax,rng,MU0,A0,BETA)
        if len(ev)<30: continue
        grid=np.linspace(0,Tmax,m); dg=grid[1]-grid[0]; K=np.zeros(m); S=0.0; idx=0
        for k in range(m):
            if k>0: S*=np.exp(-BETA*dg)
            while idx<len(ev) and ev[idx]<=grid[k]: S+=np.exp(-BETA*(grid[k]-ev[idx])); idx+=1
            K[k]=S
        lam=np.clip(MU0+A0*BETA*K,1e-9,None); s0=1.0/lam; s1=(lam-MU0)/(A0*lam); sn=np.sqrt(s0*s0+s1*s1)
        cS=[np.cumsum(lam**(2*alpha+1)*s0*s0)*dg,np.cumsum(lam**(2*alpha+1)*s0*s1)*dg,np.cumsum(lam**(2*alpha+1)*s1*s1)*dg]
        cJ=[np.cumsum(lam**(alpha+1)*s0*s0)*dg,np.cumsum(lam**(alpha+1)*s0*s1)*dg,np.cumsum(lam**(alpha+1)*s1*s1)*dg]
        an0=(lam**alpha)*sn; an5=(lam**0.5)*sn; dL0=(an0**2)*lam; dL5=(an5**2)*lam
        for i,T in enumerate(hor):
            j=min(int(T/dg),m-1); thr=eps*np.sqrt(T)
            for c in range(3): SigA[i,c]+=cS[c][j]/T; JA[i,c]+=cJ[c][j]/T
            L0[i]+=np.sum(dL0[:j+1]*(an0[:j+1]>thr))*dg/T; L5[i]+=np.sum(dL5[:j+1]*(an5[:j+1]>thr))*dg/T
        nr+=1
    SigA/=max(nr,1); JA/=max(nr,1); L0/=max(nr,1); L5/=max(nr,1)
    rng=np.random.default_rng(7); ev=simulate_hawkes(min(Tmax,1500),rng,MU0,A0,BETA)
    grid=np.linspace(0,ev.max(),m); dg=grid[1]-grid[0]; K=np.zeros(m); S=0.0; idx=0
    for k in range(m):
        if k>0: S*=np.exp(-BETA*dg)
        while idx<len(ev) and ev[idx]<=grid[k]: S+=np.exp(-BETA*(grid[k]-ev[idx])); idx+=1
        K[k]=S
    A=hawkes_A(ev,BETA); mus=np.linspace(0.4,1.8,41); as_=np.linspace(0.2,0.8,41); H=np.zeros((41,41))
    for ia,a in enumerate(as_):
        for im,mu in enumerate(mus):
            lg=np.clip(mu+a*BETA*K,1e-9,None); le=np.clip(mu+a*BETA*A,1e-9,None)
            H[ia,im]=(np.sum(lg)*dg-np.sum(np.log(le)))/ev.max()
    ja,jm=np.unravel_index(np.argmin(H),H.shape)
    fig,axs=plt.subplots(1,3,figsize=(16,4.6))
    axs[0].plot(hor,SigA[:,0],'-o',ms=3,color='tab:blue',label="T^-1 Sigma[0,0]"); axs[0].plot(hor,SigA[:,2],'-o',ms=3,color='tab:cyan',label="T^-1 Sigma[1,1]")
    axs[0].plot(hor,JA[:,0],'-s',ms=3,color='tab:red',label="T^-1 J[0,0]"); axs[0].plot(hor,JA[:,2],'-s',ms=3,color='tab:orange',label="T^-1 J[1,1]")
    axs[0].set_xlabel("T"); axs[0].set_title("(F-C3) Limites ergodiques"); axs[0].legend(fontsize=7); axs[0].grid(alpha=.3)
    axs[1].plot(hor,L5,'-o',ms=3,color='tab:green',label="alpha=0.5"); axs[1].plot(hor,L0,'-s',ms=3,color='0.5',label="alpha=0")
    axs[1].axhline(0,ls='--',color='0.6'); axs[1].set_xlabel("T"); axs[1].set_ylabel("ratio Lindeberg")
    axs[1].set_title("(F-C4) Lindeberg -> 0"); axs[1].legend(fontsize=8); axs[1].grid(alpha=.3)
    cs=axs[2].contourf(mus,as_,H,levels=25,cmap='viridis'); axs[2].plot(MU0,A0,'r*',ms=16); axs[2].plot(mus[jm],as_[ja],'wo',ms=7,mfc='none')
    axs[2].set_xlabel("mu"); axs[2].set_ylabel("a"); axs[2].set_title("(F-C2) Contraste : minimum unique"); fig.colorbar(cs,ax=axs[2],fraction=0.046)
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR,"conditions_benchmark.png"),dpi=130); plt.close(fig)


def _usgs_url(): return ("https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv"
    "&starttime=2024-01-01&endtime=2024-12-31&minmagnitude=4.5&orderby=time-asc")
_CATALOG={"seismes":dict(url=_usgs_url(),col="time",unit="days"),
          "mines":dict(url="https://vincentarelbundock.github.io/Rdatasets/csv/boot/coal.csv",col="date",unit="years")}
def _download(url,dest):
    if os.path.exists(dest): return dest
    try:
        import urllib.request
        req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0"})
        with urllib.request.urlopen(req,timeout=60) as r: data=r.read()
        open(dest,"wb").write(data); return dest
    except Exception: return None
def _load_events(jeu,datadir):
    import pandas as pd
    spec=_CATALOG[jeu]; dest=os.path.join(datadir,jeu+".csv"); path=_download(spec["url"],dest)
    if path is None: return None
    df=pd.read_csv(path); c=spec["col"]
    if spec["unit"]=="days":
        t=pd.to_datetime(df[c],errors="coerce").dropna(); ev=np.sort((t-t.min()).dt.total_seconds().values/86400.0)
    else:
        ev=np.sort(pd.to_numeric(df[c],errors="coerce").dropna().values.astype(float))
    return ev-ev.min()
def _demo_events(rng):
    mu,a,beta=2.0,0.5,1.0; t=0.0; ev=[]; last=0.0; S=0.0
    while t<200.0:
        lb=mu+a*beta*S; t+=rng.exponential(1.0/lb)
        if t>=200.0: break
        S*=np.exp(-beta*(t-last)); last=t
        if rng.uniform()<=(mu+a*beta*S)/lb: ev.append(t); S+=1.0
    return np.array(ev)
def _hawkes_nll3(th,ev,T):
    mu,a,beta=th
    if mu<=0 or a<=0 or a>=1 or beta<=0: return 1e12
    A=hawkes_A(ev,beta); li=mu+a*beta*A
    if np.any(li<=0): return 1e12
    return mu*T+a*np.sum(1-np.exp(-beta*(T-ev)))-np.sum(np.log(li))
def _hawkes_gof(ev,th):
    mu,a,beta=th; k=np.arange(len(ev)); A=hawkes_A(ev,beta); Lam=mu*ev+a*(k-A); dL=np.diff(Lam); dL=dL[dL>=0]
    return float(stats.kstest(dL,'expon').pvalue) if len(dL)>=10 else np.nan
from scipy.special import expit
def _etas_unpack(par,cmax): return (np.exp(par[0]),np.exp(par[1]),1e-4+cmax*expit(par[2]),1.0+2.0*expit(par[3]))
def _etas_lam(ev,mu,kappa,c,p):
    lam=np.full(len(ev),mu)
    for i in range(1,len(ev)): lam[i]+=kappa*np.sum((ev[i]-ev[:i]+c)**(-p))
    return lam
def _etas_comp(ev,T,mu,kappa,c,p): return mu*T+kappa/(1-p)*np.sum((T-ev+c)**(1-p)-c**(1-p))
def _etas_nll(par,ev,T,cmax):
    mu,kappa,c,p=_etas_unpack(par,cmax); lam=_etas_lam(ev,mu,kappa,c,p)
    if np.any(lam<=0) or not np.isfinite(lam).all(): return 1e12
    v=_etas_comp(ev,T,mu,kappa,c,p)-np.sum(np.log(lam)); return v if np.isfinite(v) else 1e12
def _etas_Lam(ev,mu,kappa,c,p):
    Lam=mu*ev.astype(float).copy()
    for i in range(1,len(ev)): Lam[i]+=kappa/(1-p)*np.sum((ev[i]-ev[:i]+c)**(1-p)-c**(1-p))
    return Lam
def _etas_gof(ev,th): dL=np.diff(_etas_Lam(ev,*th)); dL=dL[dL>=0]; return float(stats.kstest(dL,'expon').pvalue) if len(dL)>=10 else np.nan
def _grid_hawkes(ev,T,th,m=1500):
    mu,a,beta=th; grid=np.linspace(0,T,m); dg=grid[1]-grid[0]; K=np.zeros(m); S=0.0; idx=0
    for k in range(m):
        if k>0: S*=np.exp(-beta*dg)
        while idx<len(ev) and ev[idx]<=grid[k]: S+=np.exp(-beta*(grid[k]-ev[idx])); idx+=1
        K[k]=S
    return grid,np.clip(mu+a*beta*K,1e-300,None)

# reelles_benchmark.png
def fig_reelles(jeu="seismes"):
    datadir=os.path.join(os.path.dirname(OUTDIR),"donnees"); os.makedirs(datadir,exist_ok=True)
    ev=_load_events(jeu,datadir); note=f"{jeu} (USGS/Rdatasets)"
    if ev is None or len(ev)<30:
        ev=_demo_events(np.random.default_rng(0)); note="DEMO synthetique (pas de reseau)"
    T=ev.max()*1.0001; x=np.diff(ev); x=x[x>0]; n=len(ev)
    fams={"Weibull":stats.weibull_min,"Gamma":stats.gamma,"Lognormal":stats.lognorm}; best=None
    for fn,dist in fams.items():
        c,loc,sc=dist.fit(x,floc=0); ll=np.sum(np.log(np.clip(dist.pdf(x,c,scale=sc),1e-300,None))); aic=4-2*ll
        if best is None or aic<best[1]: best=(fn,aic)
    bf=best[0]; bdist=fams[bf]
    th0=[n/T*0.5,0.3,5.0/np.median(np.diff(ev)+1e-9)]
    hmle=optimize.minimize(_hawkes_nll3,th0,args=(ev,T),method="Nelder-Mead",options=dict(maxiter=2000,xatol=1e-4,fatol=1e-4)).x
    hg=_hawkes_gof(ev,hmle); med=np.median(np.diff(ev))+1e-9; cmax=2.0*med; ebest=None
    for p0,cf in [(1.1,0.1),(1.4,0.3)]:
        par0=[np.log(n/T*0.5),np.log(0.3),np.log(max(cf*med-1e-4,1e-6)/max(cmax-(cf*med),1e-6)),np.log((p0-1.0)/(3.0-p0))]
        r=optimize.minimize(_etas_nll,par0,args=(ev,T,cmax),method="Nelder-Mead",options=dict(maxiter=400,maxfev=600,xatol=1e-4,fatol=1e-4))
        if ebest is None or r.fun<ebest.fun: ebest=r
    eth=_etas_unpack(ebest.x,cmax); eg=_etas_gof(ev,eth)
    fig,axs=plt.subplots(1,3,figsize=(16,4.6))
    axs[0].step(ev,np.arange(1,n+1),where='post',color='0.4',lw=1)
    grid,lam=_grid_hawkes(ev,T,hmle); ax2=axs[0].twinx(); ax2.plot(grid,lam,color='tab:red',lw=0.8,alpha=0.7)
    axs[0].set_xlabel("temps"); axs[0].set_ylabel("N(t)"); ax2.set_ylabel("intensite",color='tab:red'); axs[0].set_title("Comptage + intensite Hawkes")
    mle=bdist.fit(x,floc=0); xs=np.linspace(1e-6,np.quantile(x,0.99),200)
    axs[1].hist(x,bins=40,density=True,alpha=0.6,color='tab:blue'); axs[1].plot(xs,bdist.pdf(xs,mle[0],scale=mle[2]),'r-',lw=1.5)
    axs[1].set_xlabel("inter-arrivee"); axs[1].set_ylabel("densite"); axs[1].set_title(f"Inter-arrivees + {bf}")
    def qq(dL): dq=np.sort(dL[dL>=0]); return stats.expon.ppf((np.arange(1,len(dq)+1)-0.5)/len(dq)),dq
    k=np.arange(n); A=hawkes_A(ev,hmle[2]); qh,dh=qq(np.diff(hmle[0]*ev+hmle[1]*(k-A))); qe,de=qq(np.diff(_etas_Lam(ev,*eth)))
    axs[2].plot(qh,dh,'o',ms=3,color='0.6',alpha=0.5,label=f"exp. (p={hg:.3f})"); axs[2].plot(qe,de,'o',ms=3,color='tab:green',alpha=0.6,label=f"ETAS (p={eg:.3f})")
    lim=[0,max(qh.max(),dh.max(),qe.max(),de.max())]; axs[2].plot(lim,lim,'r-')
    axs[2].set_xlabel("quantiles Exp(1)"); axs[2].set_ylabel("increments"); axs[2].set_title("GOF : exp. vs ETAS"); axs[2].legend(fontsize=8); axs[2].grid(alpha=.3)
    fig.suptitle(note); fig.tight_layout(); fig.savefig(os.path.join(OUTDIR,"reelles_benchmark.png"),dpi=130); plt.close(fig)


if __name__ == "__main__":
    fig_robustesse_renouvellement()
    fig_nonrenouvellement()
    fig_rupture()
    fig_hawkes()
    fig_conjecture1()
    fig_conditions()
    fig_reelles()
    print("figures ->", OUTDIR)
