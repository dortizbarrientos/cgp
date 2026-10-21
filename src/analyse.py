"""Generate the paper's numerical results and all figures from the equations.

Run: python src/analyse.py
All figure data are saved as CSV/JSON. PDF and SVG figures remain scalable;
PNG versions are provided for quick inspection. No empirical data are used.
"""
import json,sys,platform,time
from pathlib import Path
from dataclasses import replace,asdict
import numpy as np
import scipy
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import Ecology,episode,Fitness,GaussianSelection,access

ROOT=Path(__file__).resolve().parents[1]
RES=ROOT/'results'; FIG=ROOT/'figures'
BLUE='#2166AC'; ORANGE='#C05B2B'; GREEN='#238B45'; GRAY='#6B7280'; PURPLE='#7B4E9B'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
                     'axes.titlesize':10,'axes.spines.top':False,'axes.spines.right':False,
                     'legend.frameon':False,'legend.fontsize':8,'xtick.labelsize':8,
                     'ytick.labelsize':8,'savefig.dpi':220,'pdf.fonttype':42,'ps.fonttype':42})


def savefig(fig,name):
    for ext in ['pdf','svg','png']:
        fig.savefig(FIG/f'{name}.{ext}',bbox_inches='tight',facecolor='white')
    plt.close(fig)


def csv(name,array,header):
    np.savetxt(RES/name,np.asarray(array),delimiter=',',header=header,comments='')


def thresholds(p,mechanism='access',dt=.01):
    eps=.001
    bs=np.array([0,eps,2*eps]) if mechanism=='removal' else np.array([-eps,0,eps])
    z=episode(p,bs,mechanism=mechanism,dt=dt)['branches']
    A=z[3]+p.hybrid_value*z[4]
    if mechanism=='removal':
        S=(-3*A[0]+4*A[1]-A[2])/(2*eps); A0=A[0]
    else: S=(A[2]-A[0])/(2*eps); A0=A[1]
    R=episode(p,0,False,dt=dt)['branches'][3,0]
    D=(1-p.mating_loss)*A0-R
    # Keep roots outside [0,1]. They identify an effect with constant sign
    # over feasible probabilities, not missing data. Axes clip those roots.
    ra=D[0]/(D[0]-D[1]) if abs(D[0]-D[1])>1e-12 else np.nan
    rb=-S[0]/(S[1]-S[0]) if abs(S[1]-S[0])>1e-12 else np.nan
    return {'rho_A':float(ra),'rho_B':float(rb),'D':D,'S':S,'A':A0,'R':R}


def equilibrium(g,starts=((0,0),(-2,-1),(2,4))):
    opts=[]
    def objective(x):return -np.log(g.evaluate(x)['W'])
    for x in starts:
        o=minimize(objective,x,method='L-BFGS-B',bounds=[(-8,8),(-8,8)],
                   options={'ftol':1e-13,'gtol':1e-8,'maxiter':500})
        opts.append(o)
    o=min(opts,key=lambda x:x.fun); x=o.x; d=g.evaluate(x)
    eps=.002
    H=np.column_stack([(g.evaluate(x+np.eye(2)[j]*eps)['beta']-
                       g.evaluate(x-np.eye(2)[j]*eps)['beta'])/(2*eps) for j in range(2)])
    eig=np.linalg.eigvals(np.eye(2)+.5*g.G@H)
    return {'mean':x.tolist(),'W':d['W'],'acceptance':d['acceptance'],
            'mean_activity':d['mean_activity'],'hybrid_recruit_fraction':d['hybrid_recruit_fraction'],
            'gradient_norm':float(np.linalg.norm(d['beta'])),
            'map_eigenvalues':eig.real.tolist(),'locally_stable':bool(np.max(abs(eig))<1),
            'multistart_objective_spread':float(np.ptp([o.fun for o in opts]))}


def main():
    started=time.time(); p=Ecology(); report={'parameters':asdict(p),'illustrative_only':True}
    (RES/'parameters.json').write_text(json.dumps(asdict(p),indent=2)+'\n')
    print('Figure 1: finite egg dynamics',flush=True)
    fig,ax=plt.subplots(1,2,figsize=(7.2,3.2),layout='constrained')
    outputs=[]
    for label,b,accept,col in [('Reject',0,False,GRAY),('Accept, b = 0',0,True,ORANGE),('Accept, b = 2',2,True,BLUE)]:
        z=episode(p,b,accept,history=True)
        (t1,y1),(t2,y2)=z['history']; ts=np.r_[t1,t2]; ys=np.concatenate([y1,y2])
        ax[0].plot(ts,ys[:,0,0,1],color=col,label=label,lw=1.8)
        outputs.append(z['expected'][:,0])
        csv('fig1_'+label.split(',')[0].replace(' ','_')+f'_b{b}.csv',
            np.column_stack((ts,ys[:,0,0,1],ys[:,1,0,1],ys[:,2,0,1],ys[:,3,0,1],ys[:,4,0,1])),
            'time,eggs,C,H,C_fertilisations,H_fertilisations')
    ax[0].axvline(p.tau,color=GRAY,ls=':',lw=1);ax[0].text(p.tau+.1,92,'Later C arrives',fontsize=8)
    ax[0].set(xlabel='Time since heterospecific encounter (days)',ylabel='Unfertilised eggs',ylim=(0,105),title='A   Eggs remaining when later C arrives')
    ax[0].legend(loc='upper right',bbox_to_anchor=(1,.78))
    out=np.asarray(outputs); xs=np.arange(3)
    ax[1].bar(xs,out[:,3],color=BLUE,label='Conspecific fertilisation')
    ax[1].bar(xs,out[:,4],bottom=out[:,3],color=ORANGE,label='Heterospecific fertilisation')
    ax[1].bar(xs,out[:,0],bottom=out[:,3]+out[:,4],color='#E2E5E8',label='Unfertilised')
    ax[1].set(xticks=xs,xticklabels=['Reject','Accept\nb = 0','Accept\nb = 2'],ylabel='Eggs per initial clutch of 100',ylim=(0,112),title='B   Outcomes averaged over later receipt')
    ax[1].legend(loc='upper center',bbox_to_anchor=(.5,-.19),ncol=1)
    savefig(fig,'Figure_1_egg_dynamics')
    report['baseline_outcomes']={k:{'C_fertilised':float(v[3]),'H_fertilised':float(v[4]),'unfertilised':float(v[0]),
                                     'recruitment':float(v[3]+p.hybrid_value*v[4])}
                                 for k,v in zip(['reject','accept_b0','accept_b2'],out)}
    print('Figure 2: acceptance and restraint thresholds',flush=True)
    th=thresholds(p);report['thresholds']={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in th.items()}
    rr=np.linspace(0,1,301); D=(1-rr)*th['D'][0]+rr*th['D'][1];S=(1-rr)*th['S'][0]+rr*th['S'][1]
    csv('fig2_threshold_lines.csv',np.column_stack((rr,D,S)),'rho,acceptance_gain,restraint_derivative')
    fig=plt.figure(figsize=(7.2,5.6),layout='constrained');gs=fig.add_gridspec(2,2,height_ratios=[1,1.25])
    a=fig.add_subplot(gs[0,0]);a.plot(rr,D/p.eggs,color=ORANGE);a.axhline(0,color=GRAY,lw=.7)
    a.axvline(th['rho_A'],color=ORANGE,ls=':');a.set(xlabel='Probability of later C receipt',ylabel='Acceptance gain per egg',title='A   Benefit of receiving H')
    a=fig.add_subplot(gs[0,1]);a.plot(rr,S/p.eggs,color=BLUE);a.axhline(0,color=GRAY,lw=.7)
    a.axvline(th['rho_B'],color=BLUE,ls=':');a.set(xlabel='Probability of later C receipt',ylabel='Recruitment slope with restraint',title='B   Benefit of slowing H')
    eggs=np.geomspace(25,400,49); rows=[]
    for E in eggs:
        q=thresholds(replace(p,eggs=E));rows.append([E,q['rho_A'],q['rho_B']])
    rows=np.asarray(rows);csv('fig2_egg_thresholds.csv',rows,'eggs,rho_accept,rho_restraint')
    a=fig.add_subplot(gs[1,:]);a.plot(rows[:,0],rows[:,1],color=ORANGE,label='Acceptance boundary');a.plot(rows[:,0],rows[:,2],color=BLUE,label='Restraint boundary')
    a.fill_between(rows[:,0],rows[:,2],rows[:,1],where=rows[:,1]>rows[:,2],color='#DDEBD8',label='Acceptance and restraint both beneficial')
    a.set(xscale='log',xlabel='Initial egg number',ylabel='Probability of later C receipt',ylim=(0,1),title='C   Initial loads increase sublinearly with egg number')
    a.set_xticks([25,50,100,200,400],[25,50,100,200,400]);a.legend(loc='upper left')
    savefig(fig,'Figure_2_two_thresholds')
    print('Figure 3: mechanisms with matched initial activity',flush=True)
    fig,ax=plt.subplots(1,2,figsize=(7.2,3.4),layout='constrained');bs=np.linspace(0,5,151);mrows=[]
    mechanisms=[('access','Persistent restraint',BLUE),('removal','Initial removal',ORANGE),('early_access','Restraint until day 2',GREEN)]
    for m,label,col in mechanisms:
        vals=episode(p,bs,mechanism=m)['recruitment'];ax[0].plot(bs,vals,color=col,label=label,lw=1.8)
        for b,v in zip(bs,vals):mrows.append([m,b,float(v)])
    ax[0].axhline(float(episode(p,0,False)['recruitment'][0]),color=GRAY,ls=':',label='Reject H')
    ax[0].set(xlabel='Postmating trait b',ylabel='Expected offspring equivalents',title='A   Equal initial activity, different outcomes');ax[0].legend(fontsize=7.5)
    np.savetxt(RES/'fig3_mechanisms.csv',mrows,fmt='%s',delimiter=',',header='mechanism,b,recruitment',comments='')
    taus=np.linspace(.2,5.8,30);timing=[]
    for tau in taus:
        pp=replace(p,tau=tau)
        vals=[float(episode(pp,2,mechanism=m)['recruitment'][0]) for m,_,_ in mechanisms]
        timing.append([tau,*vals])
    timing=np.asarray(timing);csv('fig3_timing.csv',timing,'tau,persistent_restraint,initial_removal,early_restraint')
    for j,(m,label,col) in enumerate(mechanisms):ax[1].plot(taus,timing[:,j+1],color=col,label=label)
    ax[1].set(xlabel='Time of possible later C receipt (days)',ylabel='Expected offspring equivalents',title='B   Timing changes the comparison')
    savefig(fig,'Figure_3_mechanisms')
    print('Figure 4: quantitative-genetic responses and equilibria',flush=True)
    fig,ax=plt.subplots(2,2,figsize=(7.2,5.5),layout='constrained');equils={}
    for rho,col in [(.05,ORANGE),(.4,BLUE),(.9,GREEN)]:
        f=Fitness(replace(p,rho=rho));g=GaussianSelection(f)
        tr=g.trajectory(generations=30000)
        csv(f'fig4_trajectory_rho{rho}.csv',tr[::20], 'generation,r,b,mean_fitness,acceptance,hybrid_recruit_fraction,relative_H_activity,beta_r,beta_b')
        ax[0,0].plot(tr[:,0],tr[:,4],color=col,label=f'rho = {rho:g}')
        ax[0,1].plot(tr[:,0],tr[:,6],color=col,label=f'rho = {rho:g}')
        eq=equilibrium(g);eq['end_trajectory']=tr[-1].tolist()
        imin=int(np.argmin(tr[:,4]));eq['minimum_acceptance']={'generation':imin,'acceptance':float(tr[imin,4]),'hybrid_recruit_fraction':float(tr[imin,5])}
        eq['minimum_fitness_increment']=float(np.diff(tr[:,3]).min())
        equils[str(rho)]=eq
    ax[0,0].set(xlabel='Generation',ylabel='Mean H acceptance probability',title='A   Evolution of acceptance',ylim=(0,1));ax[0,0].legend()
    ax[0,1].set(xlabel='Generation',ylabel='Mean relative H activity',title='B   Evolution of postmating activity',ylim=(0,2));ax[0,1].axhline(1,color=GRAY,ls=':',lw=.8)
    eqrows=[]
    for kb,col,lab in [(.005,BLUE,'Lower restraint cost'),(.025,PURPLE,'Higher restraint cost')]:
        ar=[]
        for rho in np.linspace(0,1,21):
            g=GaussianSelection(Fitness(replace(p,rho=rho,cost_b=kb)))
            eq=equilibrium(g)
            row=[kb,rho,*eq['mean'],eq['acceptance'],eq['mean_activity'],eq['W'],eq['gradient_norm'],max(abs(np.array(eq['map_eigenvalues'])))]
            ar.append(row);eqrows.append(row)
        ar=np.asarray(ar);ax[1,0].plot(ar[:,1],ar[:,4],color=col,label=lab);ax[1,1].plot(ar[:,1],ar[:,5],color=col,label=lab)
    ax[1,0].set(xlabel='Probability of later C receipt',ylabel='Equilibrium acceptance probability',title='C   Cost changes use of premating rejection',ylim=(0,1));ax[1,0].legend()
    ax[1,1].set(xlabel='Probability of later C receipt',ylabel='Equilibrium relative H activity',title='D   Cost changes postmating restraint',ylim=(0,2));ax[1,1].axhline(1,color=GRAY,ls=':',lw=.8)
    savefig(fig,'Figure_4_evolution');csv('fig4_equilibria.csv',eqrows,'cost_b,rho,r,b,acceptance,relative_H_activity,mean_fitness,gradient_norm,map_spectral_radius')
    report['equilibria']=equils
    print('Supplementary scaling and robustness figures',flush=True)
    fig,ax=plt.subplots(1,2,figsize=(7.2,3.4),layout='constrained');sc=[]
    for j,nu in enumerate([0,1]):
      for beta,col in [(0,GRAY),(.5,BLUE),(1,GREEN)]:
        rows=[]
        for E in np.geomspace(25,400,25):
            q=thresholds(replace(p,eggs=E,beta_c=beta,beta_h=beta,beta_later=beta,volume_exponent=nu))
            rows.append([E,q['rho_A'],q['rho_B']]);sc.append([nu,beta,E,q['rho_A'],q['rho_B']])
        rows=np.asarray(rows);ax[j].plot(rows[:,0],rows[:,1],color=col,label=f'beta = {beta:g}')
        ax[j].plot(rows[:,0],rows[:,2],color=col,ls='--')
      ax[j].set(xscale='log',xlabel='Initial egg number',ylabel='Threshold probability',ylim=(0,1),title=f'{"AB"[j]}   Volume exponent = {nu}');ax[j].set_xticks([25,100,400],[25,100,400]);ax[j].legend()
    savefig(fig,'Figure_S1_scaling');csv('figS1_scaling.csv',sc,'volume_exponent,load_exponent,eggs,rho_accept,rho_restraint')
    fig,ax=plt.subplots(1,2,figsize=(7.2,3.4),layout='constrained');sens=[]
    for j,(name,xs) in enumerate([('mu_h',np.linspace(0,.8,35)),('mating_loss',np.linspace(0,.25,35))]):
        vals=[]
        for x in xs:
            q=thresholds(replace(p,**{name:float(x)}));vals.append([x,q['rho_A'],q['rho_B']]);sens.append([name,x,q['rho_A'],q['rho_B']])
        vals=np.asarray(vals);ax[j].plot(xs,vals[:,1],color=ORANGE,label='Acceptance boundary');ax[j].plot(xs,vals[:,2],color=BLUE,label='Restraint boundary')
        ax[j].fill_between(xs,vals[:,2],vals[:,1],where=vals[:,1]>vals[:,2],color='#DDEBD8')
        ax[j].set(xlabel='H functional-loss rate (per day)' if j==0 else 'Mating-associated output loss',ylabel='Threshold probability',ylim=(0,1),title=f'{"AB"[j]}   '+('Loss of H function' if j==0 else 'Cost of acceptance'))
    ax[0].legend();savefig(fig,'Figure_S2_sensitivity');np.savetxt(RES/'figS2_sensitivity.csv',sens,fmt='%s',delimiter=',',header='parameter,value,rho_accept,rho_restraint',comments='')
    # Correlated responses can reverse the initial direction of a trait.
    f=Fitness(replace(p,rho=.4));fig,ax=plt.subplots(1,2,figsize=(7.2,3.3),layout='constrained');cov=[]
    for corr,col in [(-.6,ORANGE),(0,BLUE),(.6,GREEN)]:
        G=.06*np.array([[1,corr],[corr,1]])
        g=GaussianSelection(f,G=G);tr=g.trajectory(generations=30000)
        ax[0].plot(tr[:,0],tr[:,1],color=col,label=f'Genetic correlation = {corr:g}')
        ax[1].plot(tr[:,0],tr[:,2],color=col)
        cov.append({'correlation':corr,'initial_response':g.evaluate([0,0])['response'].tolist(),'end_mean':tr[-1,1:3].tolist()})
        csv(f'figS3_covariance_{corr}.csv',tr[::20],'generation,r,b,mean_fitness,acceptance,hybrid_recruit_fraction,relative_H_activity,beta_r,beta_b')
    ax[0].set(xlabel='Generation',ylabel='Mean premating trait r',title='A   Genetic covariance changes the trajectory');ax[0].legend()
    ax[1].set(xlabel='Generation',ylabel='Mean postmating trait b',title='B   Equilibria are approached at different rates')
    savefig(fig,'Figure_S3_genetic_covariance');report['covariance_runs']=cov
    report['runtime_seconds']=time.time()-started
    report['versions']={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__}
    (RES/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Complete in',round(time.time()-started,1),'seconds',flush=True)

if __name__=='__main__':main()
