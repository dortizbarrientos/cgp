"""Recipient evolution in a gamete-persistence model (version 2).

This is a deterministic quantitative-genetic approximation, not a multilocus
speciation simulation. It retains a Gaussian phenotype distribution and fixed
P and G matrices each generation. Hybrid ancestry, linkage disequilibrium,
changes in variance, and evolution of male gamete traits are not tracked.

Three recipient traits are represented on unconstrained latent scales:
  r: heterospecific mating discrimination (positive values reduce acceptance);
  b: postmating discrimination (positive values reduce heterospecific efficacy);
  ell: log multiplier of the conspecific functional half-life.
Negative r or b represents facilitation of heterospecific contributions.

Run from any directory: python3 src/model_v2.py
All parameter values are illustrative. No empirical data are fitted.
"""
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import argparse
import itertools
import json
import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
from scipy.special import expit
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
COLOURS = ['#23758a', '#bd652b', '#73589c']


@dataclass(frozen=True)
class Parameters:
    x0: float = 4.0       # Initial conspecific availability, in scaled units.
    h0: float = 2.0       # Conspecific half-life when ell=0, in days.
    hH: float = 2.0       # Heterospecific functional half-life, in days.
    y: float = 4.0        # Functional heterospecific load at receipt.
    future_load: float = 4.0  # Conspecific load supplied at the final bout.
    tau: float = 1.0      # Days from the current encounter to fertilisation.
    rho: float = 0.1      # Probability of the later conspecific contribution.
    encounter: float = 0.6  # Chance of a current heterospecific encounter.
    w: float = 0.3        # Hybrid recruitment relative to conspecific offspring.
    c: float = 0.05       # Mating cost, paid from background C reproduction.
    background: float = 0.25  # Other prospective conspecific offspring value.
    age_min: float = 0.0
    age_max: float = 8.0
    age_mid: float = 3.0  # Age at 50% acceptance for r=0.
    age_scale: float = 1.5
    kr: float = 0.01      # Coefficients of constitutive log-fitness costs.
    kb: float = 0.01
    kl: float = 0.10


def age_grid(p, n=40):
    """Quadrature for a uniform encounter-age distribution, normalised to one."""
    nodes, weights = leggauss(n)
    ages = p.age_min + (nodes+1)*(p.age_max-p.age_min)/2
    return ages, weights/2


def phenotype_grid(mean, P, order=5):
    """Product Gauss-Hermite quadrature for N(mean,P), without random sampling."""
    nodes, weights = hermgauss(order)
    index = np.array(list(itertools.product(range(order), repeat=3)))
    standard = np.sqrt(2)*nodes[index]
    probabilities = np.prod(weights[index]/np.sqrt(np.pi), axis=1)
    phenotypes = np.asarray(mean) + standard @ np.linalg.cholesky(P).T
    return phenotypes, probabilities


def outcomes(z, ages, p):
    """Return arrays with shape (phenotype, age).

    J=0 and J=1 denote absence/presence of a later conspecific contribution.
    All focal eggs are evaluated once, after tau days. Thus no egg is counted
    twice. The model has no ongoing fertilisation or gamete consumption.
    """
    z = np.atleast_2d(z)
    r, b, ell = z.T
    h = p.h0*np.exp(ell)
    age = np.asarray(ages)[None, :]
    X = p.x0*np.exp(-np.log(2)*(age+p.tau)/h[:, None])
    Y = p.y*np.exp(-np.log(2)*p.tau/p.hH-b[:, None])
    a = expit((age-p.age_mid)/p.age_scale-r[:, None])
    R = np.zeros_like(X)
    C = np.zeros_like(X)
    H = np.zeros_like(X)
    dbA = np.zeros_like(X)
    dlR = np.zeros_like(X)
    dlA = np.zeros_like(X)
    # dX/dell: increasing log half-life preserves more of the initial store.
    dlX = X*np.log(2)*(age+p.tau)/h[:, None]
    for j, prob in ((0, 1-p.rho), (1, p.rho)):
        U = X + j*p.future_load
        D = 1+U+Y
        R += prob*U/(1+U)
        C += prob*U/D
        H += prob*Y/D
        dbA += prob*Y*((1-p.w)*U-p.w)/D**2
        dlR += prob*dlX/(1+U)**2
        dlA += prob*dlX*(1+(1-p.w)*Y)/D**2
    A = C+p.w*H
    gain = A-R-p.c
    cost = 0.5*(p.kr*r*r+p.kb*b*b+p.kl*ell*ell)
    viability = np.exp(-cost)[:, None]
    # Costs are paid by reducing OTHER conspecific reproduction. Since B>c,
    # WC and W stay positive for all phenotypes; no fitness clipping is used.
    WC = viability*(p.background+R+p.encounter*a*(C-R-p.c))
    WH = viability*p.encounter*a*p.w*H
    W = WC+WH
    # Exact phenotype derivatives of expected reproductive output.
    grad = np.stack([
        viability*(-p.encounter*a*(1-a)*gain)-p.kr*r[:, None]*W,
        viability*p.encounter*a*dbA-p.kb*b[:, None]*W,
        viability*((1-p.encounter*a)*dlR+p.encounter*a*dlA)-p.kl*ell[:, None]*W,
    ], axis=-1)
    F = R+p.encounter*a*(C+H-R)
    hybrid_zygotes = p.encounter*a*H
    return dict(W=W, WC=WC, WH=WH, a=a, gain=gain, grad=grad,
                F=F, hybrid_zygotes=hybrid_zygotes, X=X, Y=Y,
                R=R, A=A, dbA=dbA)


def population(mean, P, p, order=7, n_age=64):
    """Integrate individuals and encounter states before calculating selection.

    beta = P^{-1} Cov(z,W)/mean(W). Under a fixed Gaussian distribution it
    also equals E[gradient W]/mean(W). Evaluating W only at the population
    mean would omit the effects of trait variation near thresholds.
    """
    z, weights = phenotype_grid(mean, P, order)
    ages, age_weights = age_grid(p, n_age)
    o = outcomes(z, ages, p)
    individual_W = o['W'] @ age_weights
    mean_W = weights @ individual_W
    cov = ((z-mean).T*weights) @ individual_W
    beta = np.linalg.solve(P, cov)/mean_W
    paternal_H = weights @ (o['WH'] @ age_weights)/mean_W
    avg = lambda name: float(weights @ (o[name] @ age_weights))
    accepted = avg('a')
    control = float(expit((ages-p.age_mid)/p.age_scale) @ age_weights)
    # A standardised forced-mating assay: functional C=H=1, no ageing,
    # fertilisation scored before offspring mortality. Pool counts, not ratios.
    q = np.exp(-z[:, 1])
    assay_C = 1/(2+q)
    assay_H = q/(2+q)
    assay_PC = float(weights@assay_C / (weights@(assay_C+assay_H)))
    return dict(mean_W=float(mean_W), beta=beta, paternal_H=float(paternal_H),
                acceptance=accepted, premating=1-accepted/control,
                assay_PC=assay_PC, hybrid_share=avg('hybrid_zygotes')/avg('F'),
                fertilisation=avg('F'),
                occupancy=float(weights @ ((o['gain']>0) @ age_weights)),
                mean_h=float(weights@(p.h0*np.exp(z[:, 2]))),
                gradient_beta=np.einsum('i,ijk,j->k',weights,o['grad'],age_weights)/mean_W)


def trajectory(p, P, G, generations=600, order=7, n_age=64, gene_input=True):
    """Iterate a fixed-covariance quantitative-genetic mean recursion.

    The one-half factors follow from biparental autosomal inheritance with
    selection expressed through recipients. Conspecific fathers are sampled
    independently with the current resident mean. Heterospecific fathers have
    fixed latent recipient-trait breeding-value mean zero. Their actual
    contribution is calculated from viable recruits, not raw mating rate.

    Replacing the offspring distribution by N(new_mean,P) is an APPROXIMATION.
    This is not a claim that actual hybrid offspring are normally distributed.
    """
    mean = np.zeros(3)
    external_mean = np.zeros(3)
    rows = []
    for generation in range(generations+1):
        s = population(mean, P, p, order, n_age)
        selection = 0.5*G@s['beta']
        incoming = 0.5*s['paternal_H']*(external_mean-mean) if gene_input else np.zeros(3)
        rows.append([generation,*mean,s['mean_h'],s['mean_W'],*s['beta'],
                     *selection,*incoming,s['paternal_H'],s['acceptance'],
                     s['premating'],s['assay_PC'],s['hybrid_share'],
                     s['fertilisation'],s['occupancy']])
        mean = mean+selection+incoming
    return np.asarray(rows)


FIELDS = ['generation','mean_r','mean_b','mean_ell','mean_half_life_days',
          'mean_fitness','beta_r','beta_b','beta_ell','selection_r','selection_b',
          'selection_ell','incoming_r','incoming_b','incoming_ell',
          'heterospecific_paternal_fraction','acceptance','premating_index',
          'standardised_conspecific_paternity','hybrid_zygote_share',
          'terminal_fertilisation','beneficial_acceptance_state_fraction']


def checks(p, P, G):
    """Check biological limits, analytical derivatives and integration accuracy."""
    assert p.background>p.c>=0
    assert np.linalg.eigvalsh(G).min()>0
    assert np.linalg.eigvalsh(P-G).min()>=0
    rng = np.random.default_rng(41)
    z = rng.normal(0,0.4,(9,3))
    ages = np.array([0.0,2.0,5.0,8.0])
    o = outcomes(z,ages,p)
    eps=1e-5
    fd=[]
    for k in range(3):
        v=np.zeros(3);v[k]=eps
        fd.append((outcomes(z+v,ages,p)['W']-outcomes(z-v,ages,p)['W'])/(2*eps))
    error=float(np.max(np.abs(np.stack(fd,axis=-1)-o['grad'])))
    assert error<1e-8
    assert np.all(o['WC']>0) and np.all(o['WH']>=0)
    assert np.all((o['F']>=0)&(o['F']<=1))
    # Exact recovery of the original marginal payoff (rho=tau=b=0).
    old=replace(p,rho=0,tau=0)
    legacy=outcomes(np.zeros(3),ages,old)
    x=p.x0*2**(-ages/p.h0)
    delta=p.y/(1+x+p.y)*(p.w-x/(1+x))-p.c
    legacy_error=float(np.max(np.abs(legacy['gain'][0]-delta)))
    assert legacy_error<1e-14
    # No heterospecific encounter: neither r nor b has a direct benefit.
    absent=population(np.zeros(3),P,replace(p,encounter=0),7,64)
    assert np.max(np.abs(absent['beta'][:2]))<1e-10
    # w=1: suppressing H efficacy cannot increase the direct reproductive payoff.
    assert np.all(outcomes(z,ages,replace(p,w=1))['dbA']<0)
    # Direct gradients and Gaussian covariance gradients are independent checks.
    a=population(np.array([0.3,0.2,0.1]),P,p,7,64)
    grad_error=float(np.max(np.abs(a['beta']-a['gradient_beta'])))
    assert grad_error<2e-5
    return dict(finite_difference_gradient_max_error=error,
                version_1_recovery_max_error=legacy_error,
                gaussian_gradient_identity_max_error=grad_error,
                positive_fitness=True,probability_bounds=True,
                no_encounter_selection_limit=True,w_equals_1_limit=True)


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'axes.titleweight':'bold','axes.titlesize':11,
                         'axes.labelsize':10,'legend.frameon':False,
                         'savefig.dpi':220,'pdf.fonttype':42,'svg.fonttype':'none'})


def save_figure(fig, root, name):
    for extension in ('pdf','svg','png'):
        fig.savefig(root/'figures'/f'{name}.{extension}',bbox_inches='tight',facecolor='white')
    plt.close(fig)


def figures(root,p,traces):
    style()
    fig,ax=plt.subplots(1,2,figsize=(11.4,4.1),layout='constrained')
    ages=np.linspace(0,12,401)
    for colour,rho in zip(COLOURS,[0.0,0.5,1.0]):
        o=outcomes(np.zeros(3),ages,replace(p,rho=rho))
        ax[0].plot(ages,o['gain'][0],color=colour,label=f'Future C probability {rho:g}')
    ax[0].axhline(0,color='#606870',lw=.8)
    ax[0].set(xlabel='Age at heterospecific encounter (days)',ylabel='Accept minus reject payoff',title='A  Future conspecific supply changes acceptance')
    ax[0].legend(fontsize=8,loc='upper left')
    X=np.linspace(0,5,400);Y=2.;w=p.w
    suppress=Y*((1-w)*X-w)/(1+X+Y)**2
    redistribute=(X+Y)/(1+X+Y)*(1-w)*X*Y/(X+Y)**2
    ax[1].plot(X,suppress,color=COLOURS[0],label='Suppress H efficacy: fertility can fall')
    ax[1].plot(X,redistribute,color=COLOURS[1],ls='--',label='Redistribute paternity: fertility fixed')
    ax[1].axhline(0,color='#606870',lw=.8)
    threshold=w/(1-w)
    ax[1].axvline(threshold,color='#88929a',lw=.8,ls=':')
    ax[1].set(xlabel='Conspecific availability at fertilisation, U',ylabel='Direct payoff derivative with respect to b',title='B  The mechanism changes selection on bias')
    ax[1].legend(fontsize=8,loc='lower right')
    fig.suptitle('1  Gamete availability changes the value of reproductive barriers',fontsize=14,fontweight='bold')
    save_figure(fig,root,'Figure_1_Decisions_and_Mechanisms')

    fig,ax=plt.subplots(1,3,figsize=(11.4,3.9),layout='constrained')
    for i,(label,tr) in enumerate(traces.items()):
        for axis,col in zip(ax,[1,2,4]):
            axis.plot(tr[:,0],tr[:,col],color=COLOURS[i],label=label)
    for axis,title,ylabel in zip(ax,['A  Premating discrimination','B  Postmating discrimination','C  Conspecific persistence'],['Mean r','Mean b','Mean functional half-life (days)']):
        axis.set(xlabel='Generation',ylabel=ylabel,title=title)
    ax[0].legend(fontsize=8)
    fig.suptitle('2  Trait trajectories depend on future compatible supply',fontsize=14,fontweight='bold')
    save_figure(fig,root,'Figure_2_Recipient_Evolution')

    fig,ax=plt.subplots(2,2,figsize=(10.8,7.0),layout='constrained')
    for i,(label,tr) in enumerate(traces.items()):
        for axis,col in zip(ax.flat,[16,17,18,19]):
            axis.plot(tr[:,0],tr[:,col],color=COLOURS[i],label=label)
    labels=[('A  Heterospecific acceptance','Probability conditional on encounter'),
            ('B  Premating isolation index','1 - H acceptance / C acceptance'),
            ('C  Standardised mixed-gamete assay','Conspecific share among fertilised eggs'),
            ('D  Realised hybrid fertilisation','Hybrid share in the terminal bout')]
    for axis,(title,ylabel) in zip(ax.flat,labels):
        axis.set(xlabel='Generation',ylabel=ylabel,title=title)
    ax[0,0].legend(fontsize=8)
    ax[1,0].axhline(.5,color='#88929a',lw=.8,ls=':')
    fig.suptitle('3  Acceptance, measured CGP and hybridisation are different outcomes',fontsize=14,fontweight='bold')
    save_figure(fig,root,'Figure_3_Isolation_Outcomes')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generations',type=int,default=600)
    parser.add_argument('--output',type=Path,default=ROOT)
    args=parser.parse_args();root=args.output.resolve()
    for directory in ['figures','data']:(root/directory).mkdir(parents=True,exist_ok=True)
    p=Parameters();P=np.diag([.16,.16,.16]);G=np.diag([.08,.08,.03])
    diagnostics=checks(p,P,G)
    traces={};summaries={}
    for label,rho in [('Low future C supply (0.1)',.1),('High future C supply (0.8)',.8)]:
        scenario=replace(p,rho=rho)
        tr=trajectory(scenario,P,G,args.generations)
        traces[label]=tr
        stem=f'future_C_{rho:.1f}'.replace('.','p')
        np.savetxt(root/'data'/f'{stem}.csv',tr,delimiter=',',header=','.join(FIELDS),comments='')
        summaries[label]={name:float(value) for name,value in zip(FIELDS,tr[-1])}
        # A finer quadrature trajectory checks accumulated numerical error.
        fine=trajectory(scenario,P,G,args.generations,order=9,n_age=80)
        accumulated=float(np.max(np.abs(tr[:,:6]-fine[:,:6])))
        assert accumulated<2e-4, accumulated
        diagnostics[f'{stem}_trajectory_quadrature_error']=accumulated
    # Sensitivity to genetic covariance and paternal gene input, without a new
    # figure: these are controls, not additional fitted scenarios.
    negG=G.copy();negG[0,1]=negG[1,0]=-.8*np.sqrt(G[0,0]*G[1,1])
    for label,g,flow in [('negative_genetic_correlation',negG,True),('selection_only',G,False)]:
        tr=trajectory(p,P,g,args.generations,gene_input=flow)
        np.savetxt(root/'data'/f'{label}.csv',tr,delimiter=',',header=','.join(FIELDS),comments='')
        summaries[label]={name:float(value) for name,value in zip(FIELDS,tr[-1])}
        if label=='selection_only':
            traces['Selection-only reference (0.1)']=tr
            assert np.min(np.diff(tr[:,5]))>-1e-8
            assert abs(tr[-1,2]-traces['Low future C supply (0.1)'][-1,2])>1e-3
            diagnostics['selection_only_counterfactual_distinct']=True
            diagnostics['selection_only_fitness_nondecreasing']=True
            fine=trajectory(p,P,g,args.generations,order=9,n_age=80,gene_input=False)
            err=float(np.max(np.abs(tr[:,:6]-fine[:,:6])))
            assert err<2e-4, err
            diagnostics['selection_only_trajectory_quadrature_error']=err
    config={'parameters':asdict(p),'P':P.tolist(),'G':G.tolist(),
            'external_trait_mean':[0,0,0],'generations':args.generations,
            'integration':'7-point Gauss-Hermite per trait, 64-point age quadrature',
            'high_resolution_check':'9-point Gauss-Hermite, 80-point age quadrature',
            'seed':'No stochastic simulation; seed 41 used only for derivative checks.'}
    (root/'parameters.json').write_text(json.dumps(config,indent=2)+'\n')
    (root/'checks.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    (root/'results.json').write_text(json.dumps(summaries,indent=2)+'\n')
    figures(root,p,traces)
    print(json.dumps({'checks':diagnostics,'endpoints':summaries},indent=2))


if __name__=='__main__':main()
