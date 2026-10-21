"""Independent numerical and analytical checks for the manuscript.

Run from the package root: python src/validate.py
The report records errors as well as pass/fail flags. A failed assertion stops
the run; no failed check is silently removed from the output.
"""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import expit
from dataclasses import replace
from model import Ecology, episode, Fitness, GaussianSelection, access

ROOT=Path(__file__).resolve().parents[1]


def independent(p,b,accept=True,future=True,adjoint=False):
    """Adaptive DOP853 solution, independently written from the RK4 RHS."""
    C,H,L=p.loads(); H=H if accept else 0.; kc=p.rate; kh=kc*access(b)
    def fun(t,y):
        E,C,H=y[:3]; fc=kc*C*E; fh=kh*H*E
        return [-fc-fh,-fc-p.mu_c*C,-fh-p.mu_h*H,fc,fh,
                p.mu_c*C,p.mu_h*H]
    y0=[p.eggs,C,H,0,0,0,0]
    a=solve_ivp(fun,[0,p.tau],y0,method='DOP853',rtol=1e-11,atol=1e-12,dense_output=True)
    y1=a.y[:,-1].copy(); y1[1]+=L if future else 0.
    d=solve_ivp(fun,[p.tau,p.end],y1,method='DOP853',rtol=1e-11,atol=1e-12,dense_output=True)
    result={'end':d.y[:,-1], 'segments':(a,d)}
    if not adjoint: return result
    def reverse(sol):
        def adj(t,v):
            E,C,H=sol.sol(t)[:3]; pe,pc,ph=v[:3]
            # These are -L_x-J_F^T p, not a time reversal of the state RHS.
            dpE=-(kc*C+p.hybrid_value*kh*H)+(kc*C+kh*H)*pe+kc*C*pc+kh*H*ph
            dpC=-kc*E+kc*E*pe+(p.mu_c+kc*E)*pc
            dpH=-p.hybrid_value*kh*E+kh*E*pe+(p.mu_h+kh*E)*ph
            q=-expit(b)*kh*E*H*(p.hybrid_value-pe-ph)
            return [dpE,dpC,dpH,-q]
        return adj
    v=np.zeros(4)
    for sol,left,right in [(d,p.tau,p.end),(a,0,p.tau)]:
        z=solve_ivp(reverse(sol),[right,left],v,method='DOP853',rtol=1e-10,atol=1e-11)
        v=z.y[:,-1]
    result['adjoint']=v
    return result


def stochastic(p,b,n=5000,future=False,seed=20261020):
    """Vectorised exact Gillespie simulation for integer initial counts.

    Each row follows its own event clock. The only scheduled jump is later C
    receipt. Events are C fertilisation, H fertilisation, C loss and H loss.
    The fixed seed and sample size are recorded in the validation report.
    """
    rng=np.random.default_rng(seed)
    C,H,L=p.loads()
    if any(abs(v-round(v))>1e-9 for v in [p.eggs,C,H,L]):
        raise ValueError('Stochastic validation needs integer counts')
    E=np.full(n,int(p.eggs)); C=np.full(n,int(C)); H=np.full(n,int(H))
    ZC=np.zeros(n); ZH=np.zeros(n); t=np.zeros(n); arrived=np.zeros(n,bool)
    active=np.ones(n,bool); kh=p.rate*access(b)
    while np.any(active):
        idx=np.flatnonzero(active)
        rates=np.column_stack((p.rate*E[idx]*C[idx],kh*E[idx]*H[idx],
                               p.mu_c*C[idx],p.mu_h*H[idx]))
        total=rates.sum(axis=1)
        wait=np.full(len(idx),np.inf)
        positive=total>0
        wait[positive]=rng.exponential(1/total[positive])
        scheduled=np.where(~arrived[idx],p.tau,p.end)
        event_time=t[idx]+wait
        jump=event_time>=scheduled
        ji=idx[jump]; t[ji]=scheduled[jump]
        arrival=~arrived[ji]
        ai=ji[arrival]; arrived[ai]=True
        if future: C[ai]+=int(L)
        active[ji[~arrival]]=False
        ei=idx[~jump]; t[ei]=event_time[~jump]
        if len(ei):
            cumulative=np.cumsum(rates[~jump],axis=1)
            draws=rng.random(len(ei))*total[~jump]
            ev=np.sum(draws[:,None]>=cumulative,axis=1)
            for j in range(4):
                ii=ei[ev==j]
                if j==0: E[ii]-=1; C[ii]-=1; ZC[ii]+=1
                elif j==1: E[ii]-=1; H[ii]-=1; ZH[ii]+=1
                elif j==2: C[ii]-=1
                else: H[ii]-=1
    W=ZC+p.hybrid_value*ZH
    return {'mean':float(W.mean()),'se':float(W.std(ddof=1)/np.sqrt(n)),
            'n':n,'seed':seed}


def run():
    checks={}; p=Ecology()
    max_conservation=max_negative=max_solver=max_step=0.
    for E in [25,100,400]:
      for mu in [0,.08,.4]:
        pp=replace(p,eggs=E,mu_c=mu,mu_h=mu)
        for b in [-2,0,3]:
            a=episode(pp,b,dt=.005)['branches'][:,0]
            coarse=episode(pp,b,dt=.01)['branches'][:,0]
            max_step=max(max_step,float(np.max(np.abs(coarse-a))))
            C,H,L=pp.loads()
            max_conservation=max(max_conservation,float(np.max(np.abs(a[0]+a[3]+a[4]-E))),
                                 float(np.max(np.abs(a[1]+a[3]+a[5]-np.array([C,C+L])))),
                                 float(np.max(np.abs(a[2]+a[4]+a[6]-H))))
            max_negative=min(max_negative,float(a.min()))
            for future in [False,True]:
                z=independent(pp,b,future=future)['end']
                max_solver=max(max_solver,float(np.max(np.abs(z-a[:,int(future)]))))
    checks['conservation_max_absolute_error']=max_conservation
    checks['minimum_state']=max_negative
    checks['RK4_vs_DOP853_max_absolute_error']=max_solver
    checks['production_step_001_vs_0005_max_absolute_error']=max_step
    assert max_conservation<1e-8 and max_negative>-1e-8 and max_solver<1e-4
    assert max_step<1e-4
    # Equal initial E and C, no H and no mortality has an exact solution.
    pp=replace(p,c_ref=100,h_ref=0,later_ref=0,mu_c=0,mu_h=0)
    z=episode(pp,0)['branches'][:,0,0]
    expected=100/(1+pp.rate*100*pp.end)
    checks['exact_equal_pool_error']=float(abs(z[0]-expected))
    assert checks['exact_equal_pool_error']<1e-5
    # The generic model must give zero fertilisation without sperm.
    pp=replace(p,c_ref=0,h_ref=0,later_ref=0)
    z=episode(pp,0)['expected'][:,0]
    assert z[3]==z[4]==0 and z[0]==p.eggs
    checks['zero_sperm_limit']=True
    # Validate the time-dependent marginal-value calculation independently.
    errors=[]; eps=1e-4
    for b in [0,1.5,3]:
      for future in [False,True]:
        a=independent(p,b,future=future,adjoint=True)
        plus=independent(p,b+eps,future=future)['end']
        minus=independent(p,b-eps,future=future)['end']
        fd=((plus[3]+p.hybrid_value*plus[4])-(minus[3]+p.hybrid_value*minus[4]))/(2*eps)
        errors.append(abs(fd-a['adjoint'][3]))
    checks['adjoint_derivative_max_error']=float(max(errors))
    assert max(errors)<2e-5
    # Exact scaling symmetry when all loads and volume scale linearly with E.
    vals=[]
    for E in [25,100,400]:
        pp=replace(p,eggs=E,beta_c=1,beta_h=1,beta_later=1,volume_exponent=1)
        vals.append(episode(pp,1)['expected'][:,0]/E)
    checks['linear_scaling_symmetry_error']=float(np.ptp(vals,axis=0).max())
    assert checks['linear_scaling_symmetry_error']<1e-10
    # Generalized Bernoulli thresholds equal the zeros obtained directly.
    a=episode(p,[-eps,0,eps])['branches']
    A=a[3]+p.hybrid_value*a[4]
    R=episode(p,0,False)['branches'][3,0]
    D=(1-p.mating_loss)*A[1]-R; S=(A[2]-A[0])/(2*eps)
    rhoA=D[0]/(D[0]-D[1]); rhoB=-S[0]/(S[1]-S[0])
    checks['baseline_thresholds']={'rho_accept':float(rhoA),'rho_restraint':float(rhoB),
                                  'D_branches':D.tolist(),'S_branches':S.tolist()}
    assert 0<rhoB<rhoA<1
    # The analytical limiting-model interval must exist throughout this grid.
    gaps=[]
    for w in [.05,.4,.95]:
      for lam in [.01,.2,2,10]:
       for tau in [.2,2,5.9]:
        T=6.; d0=w*(-np.expm1(-lam*T)); d1=-(1-w)*(-np.expm1(-lam*tau))
        rb=w*T*np.exp(-lam*T)/(w*T*np.exp(-lam*T)+(1-w)*tau*np.exp(-lam*tau))
        ra=d0/(d0-d1); gaps.append(ra-rb)
    checks['limiting_model_minimum_threshold_gap']=float(min(gaps))
    assert min(gaps)>0
    # Gaussian quadrature, selection gradient and independent numerical slope.
    f=Fitness(replace(p,cost_b=.005),dt=.005)
    gs=[GaussianSelection(f,order=o) for o in [7,9,13]]
    mean=np.array([-.6,2.]); vals=[g.evaluate(mean) for g in gs]
    checks['quadrature_W_relative_error_9_vs_13']=abs(vals[1]['W']/vals[2]['W']-1)
    fd=np.array([(np.log(gs[1].evaluate(mean+np.eye(2)[j]*eps)['W'])-
                  np.log(gs[1].evaluate(mean-np.eye(2)[j]*eps)['W']))/(2*eps) for j in range(2)])
    checks['selection_gradient_vs_mean_fitness_slope']=float(np.max(abs(fd-vals[1]['beta'])))
    assert checks['quadrature_W_relative_error_9_vs_13']<1e-6
    assert checks['selection_gradient_vs_mean_fitness_slope']<5e-5
    # Matched initial activity: removal and rate restraint diverge later.
    z={m:float(episode(p,2,mechanism=m)['recruitment'][0]) for m in ['access','removal','early_access']}
    checks['matched_initial_activity_recruitment']=z
    assert z['early_access']>z['access']>z['removal']
    # Stochastic checks report discrepancies, not a false identity of ODE means.
    st={}
    for b in [0.,2.]:
        branches=[stochastic(p,b,n=5000,future=j,seed=20261020+100*int(b)+int(j)) for j in [False,True]]
        mean=(1-p.rho)*branches[0]['mean']+p.rho*branches[1]['mean']
        se=np.sqrt(((1-p.rho)*branches[0]['se'])**2+(p.rho*branches[1]['se'])**2)
        det=float(episode(p,b)['recruitment'][0])
        st[str(b)]={'branches':branches,'expected_mean':mean,'se':float(se),'deterministic':det,
                    'relative_difference':float((mean-det)/det)}
    checks['Gillespie']=st
    assert st['2.0']['expected_mean']>st['0.0']['expected_mean']
    checks['all_checks_passed']=True
    (ROOT/'results'/'validation.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))
    return checks

if __name__=='__main__': run()
