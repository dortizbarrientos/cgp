"""Finite-egg reproductive episodes and recipient-trait evolution.

All quantities are illustrative. C and H count sperm that remain functional;
their loss rates therefore combine irreversible functional failure and death.
The main model does not claim to identify those biological processes.

The vectorised RK4 solver integrates both donor contributions, unfertilised
eggs, fertilised eggs, and losses. A separate SciPy solver in validate.py
checks it. Future conspecific receipt is a Bernoulli event whose two branches
are integrated explicitly; an average load is never substituted for it.
"""
from dataclasses import dataclass, asdict, replace
import numpy as np
from scipy.special import expit
from numpy.polynomial.hermite import hermgauss


@dataclass(frozen=True)
class Ecology:
    eggs: float = 100.
    c_ref: float = 8.
    h_ref: float = 100.
    later_ref: float = 250.
    egg_ref: float = 100.
    beta_c: float = .5
    beta_h: float = .5
    beta_later: float = .5
    volume_exponent: float = 1.
    age: float = 0.
    mu_c: float = .08
    mu_h: float = .08
    k_ref: float = .02
    tau: float = 2.
    end: float = 6.
    rho: float = .4
    hybrid_value: float = .4
    mating_loss: float = .02
    cost_r: float = .025
    cost_b: float = .005
    acceptance_offset: float = 0.

    def loads(self):
        """Scaling is applied to INITIAL egg number, never remaining eggs."""
        scale = self.eggs / self.egg_ref
        return (self.c_ref * scale**self.beta_c * np.exp(-self.mu_c*self.age),
                self.h_ref * scale**self.beta_h,
                self.later_ref * scale**self.beta_later)

    @property
    def rate(self):
        return self.k_ref*(self.eggs/self.egg_ref)**(-self.volume_exponent)


def access(b):
    """Relative H activity: bounded above by 2, equals 1 at b=0."""
    return 2.*expit(-np.asarray(b))


def rhs(y, k_c, k_h, mu_c, mu_h):
    """Rows: E, C, H, Z_C, Z_H, loss_C, loss_H. Rates are per day."""
    E, C, H = y[:3]
    f_c, f_h = k_c*E*C, k_h*E*H
    return np.stack((-f_c-f_h, -mu_c*C-f_c, -mu_h*H-f_h,
                     f_c, f_h, mu_c*C, mu_h*H))


def integrate_segment(y, duration, k_c, k_h, mu_c, mu_h, dt=.01,
                      keep=False, offset=0.):
    """Classical RK4 with a shortened step so segment boundaries are exact."""
    if duration == 0:
        return y, (np.array([offset]), y[None].copy()) if keep else None
    steps = int(np.ceil(duration/dt)); h = duration/steps
    history = [y.copy()] if keep else None
    for _ in range(steps):
        d1=rhs(y,k_c,k_h,mu_c,mu_h)
        d2=rhs(y+h*d1/2,k_c,k_h,mu_c,mu_h)
        d3=rhs(y+h*d2/2,k_c,k_h,mu_c,mu_h)
        d4=rhs(y+h*d3,k_c,k_h,mu_c,mu_h)
        y=y+h*(d1+2*d2+2*d3+d4)/6
        if keep: history.append(y.copy())
    hist=(offset+np.linspace(0,duration,steps+1),np.asarray(history)) if keep else None
    return y,hist


def episode(ecology=Ecology(), b=0., accept=True, mechanism='access',
            dt=.01, history=False):
    """Return both future-arrival branches and their expected recruitment.

    mechanism='access': change H's fertilisation rate throughout the episode.
    mechanism='removal': reduce H's initial number by the same multiplier;
       compare b>=0 only, so this represents removal rather than addition.
    mechanism='early_access': change H access only before the later-arrival
       time; release the restraint thereafter, whether or not C arrives.

    b can be a vector. The final branch axis is [no later C, later C].
    Mortality is independent of b in all three mechanisms.
    """
    p=ecology; b=np.atleast_1d(np.asarray(b,dtype=float)); n=len(b)
    if not (0<=p.tau<=p.end): raise ValueError('Need 0 <= tau <= end')
    if mechanism=='removal' and np.any(b<0):
        raise ValueError('Removal comparisons require b >= 0')
    C,H,L=p.loads(); factors=access(b)
    initial_h=np.full(n,H if accept else 0.)
    if mechanism=='removal': initial_h*=factors
    y=np.zeros((7,n,2)); y[0]=p.eggs; y[1]=C
    y[2]=initial_h[:,None]
    kh=p.rate*(factors if mechanism!='removal' else np.ones(n))[:,None]
    y,h1=integrate_segment(y,p.tau,p.rate,kh,p.mu_c,p.mu_h,dt,history)
    before=y.copy(); y[1,:,1]+=L
    if mechanism=='early_access': kh=np.full((n,1),p.rate)
    y,h2=integrate_segment(y,p.end-p.tau,p.rate,kh,p.mu_c,p.mu_h,dt,history,p.tau)
    weights=np.array([1-p.rho,p.rho])
    expected=np.sum(y*weights,axis=-1)
    out={'branches':y,'expected':expected,'before_arrival':before,
         'recruitment':expected[3]+p.hybrid_value*expected[4],
         'initial_h':initial_h,'b':b}
    if history: out['history']=(h1,h2)
    return out


class Fitness:
    """Fast fitness evaluation using an accurately tabulated episode curve.

    Interpolation is used only after solving the two physiological branches.
    r determines acceptance, b determines the conditional postmating episode.
    Expected FITNESS is averaged over arrival uncertainty, not log fitness.
    """
    def __init__(self,p=Ecology(),dt=.01,b_grid=None,mechanism='access'):
        from scipy.interpolate import PchipInterpolator
        self.p=p
        self.grid=np.linspace(-12,12,1201) if b_grid is None else b_grid
        a=episode(p,self.grid,True,mechanism,dt)
        self.accepted=a
        self.A=PchipInterpolator(self.grid,a['recruitment'],extrapolate=False)
        self.ZC=PchipInterpolator(self.grid,a['expected'][3],extrapolate=False)
        self.ZH=PchipInterpolator(self.grid,a['expected'][4],extrapolate=False)
        self.R=float(episode(p,0.,False,dt=dt)['recruitment'][0])
        self.mechanism=mechanism

    def components(self,z):
        z=np.asarray(z); r,b=z[...,0],z[...,1]
        a=expit(self.p.acceptance_offset-r)
        if np.any(b<self.grid[0]) or np.any(b>self.grid[-1]):
            raise ValueError('Trait outside interpolation range; extend grid')
        survival=np.exp(-.5*(self.p.cost_r*r*r+self.p.cost_b*b*b))
        C=survival*((1-a)*self.R+a*(1-self.p.mating_loss)*self.ZC(b))
        H=survival*a*(1-self.p.mating_loss)*self.p.hybrid_value*self.ZH(b)
        return C,H,a

    def __call__(self,z):
        C,H,_=self.components(z)
        return C+H

    def gradient_log(self,z):
        """Exact r derivative and derivative of the interpolated A(b)."""
        r,b=np.asarray(z); p=self.p; a=expit(p.acceptance_offset-r)
        D=(1-p.mating_loss)*self.A(b)-self.R
        base=self.R+a*D
        return np.array([-p.cost_r*r-a*(1-a)*D/base,
                         -p.cost_b*b+a*(1-p.mating_loss)*self.A.derivative()(b)/base])


class GaussianSelection:
    """Fixed-covariance Gaussian quantitative-genetic closure.

    P is phenotypic covariance and G is additive genetic covariance. Selection
    acts through recipients; unselected sires have the resident mean. Therefore
    the response is one-half G beta. The distribution is approximated as normal
    again each generation, with P and G fixed. This is NOT an explicit hybrid
    zone or a simulation of ancestry. Hermite quadrature integrates W itself.
    """
    def __init__(self,fitness,P=None,G=None,order=9):
        self.fitness=fitness
        self.P=np.eye(2)*.12 if P is None else np.asarray(P)
        self.G=np.eye(2)*.06 if G is None else np.asarray(G)
        self.order=order
        x,w=hermgauss(order)
        xx,yy=np.meshgrid(x,x,indexing='ij')
        self.offsets=np.sqrt(2)*np.column_stack((xx.ravel(),yy.ravel()))@np.linalg.cholesky(self.P).T
        self.weights=np.outer(w,w).ravel()/np.pi

    def evaluate(self,mean):
        z=np.asarray(mean)+self.offsets
        C,H,a=self.fitness.components(z); W=C+H
        meanW=float(self.weights@W)
        cov=np.sum(self.weights[:,None]*self.offsets*W[:,None],axis=0)
        beta=np.linalg.solve(self.P,cov)/meanW
        return {'W':meanW,'beta':beta,'response':.5*self.G@beta,
                'acceptance':float(self.weights@a),
                'hybrid_recruit_fraction':float(self.weights@H/meanW),
                'mean_activity':float(self.weights@access(z[:,1]))}

    def trajectory(self,start=(0.,0.),generations=1500):
        mean=np.asarray(start,dtype=float); records=[]
        for g in range(generations+1):
            d=self.evaluate(mean)
            records.append([g,*mean,d['W'],d['acceptance'],
                            d['hybrid_recruit_fraction'],d['mean_activity'],
                            *d['beta']])
            mean=mean+d['response']
        return np.asarray(records)
