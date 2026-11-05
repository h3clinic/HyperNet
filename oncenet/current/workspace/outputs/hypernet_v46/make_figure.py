"""Plot the audited real-DEV duration/generator factorial, with paired seeds.

Reads frozen outputs only. It performs no inference, tuning, or significance test.
"""
from pathlib import Path
import hashlib,json
import numpy as np
R=Path(__file__).resolve().parent
SEEDS=(45011,45029,45047)
GROUPS=('random','shapes','spectrogram')
LABELS=('Random weights','Prior shape training','Spectrogram training')
VIEWS=('single','multi')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_audited():
    result_path=R/'evaluation/results.json';audit_path=R/'evaluation/audit.json'
    synthetic_path=R/'spectrogram/synthetic_results.json';synthetic_audit_path=R/'synthetic_audit.json'
    paths=[result_path,audit_path,synthetic_path,synthetic_audit_path,R/'evaluation/protocol.json',R/'spectrogram/protocol.json',R/'data/audit.json',R/'evaluation_data_preflight.json',R/'data/collision_review.json']
    missing=[str(p)for p in paths if not p.exists()]
    if missing:raise RuntimeError('Wait for completed results and audits: '+', '.join(missing))
    result=read(result_path);audit=read(audit_path);synthetic_audit=read(synthetic_audit_path)
    assert result['complete']and result['no_selection']and result['no_new_real_holdout']
    assert audit['passed']and audit['results_sha256']==sha(result_path)
    assert audit['protocol_sha256']==result['protocol_sha256']==sha(R/'evaluation/protocol.json')
    assert synthetic_audit['passed']and synthetic_audit['results_sha256']==sha(synthetic_path)
    assert synthetic_audit['protocol_sha256']==sha(R/'spectrogram/protocol.json')and synthetic_audit['model_count']==9 and synthetic_audit['set_count']==2
    preflight=read(R/'evaluation_data_preflight.json')
    assert preflight['passed'] and preflight['no_model_scores']
    assert all(sha(p)==digest for p,digest in preflight['files'].items())
    assert audit['data']['unsafe_cross_role_collisions']==0
    cells={key:np.empty((3,2,3),dtype=float)for key in ['pauc_std_01','recall','fa_per10000','mcc']}
    for gi,group in enumerate(GROUPS):
        for vi,view in enumerate(VIEWS):
            for si,seed in enumerate(SEEDS):
                matches=[x for x in result['macro']if x['training']==group and x['seed']==seed and x['view']==view and x['alpha']==.001]
                assert len(matches)==1,(group,view,seed,len(matches));row=matches[0]
                for key in cells:cells[key][gi,vi,si]=row[key]*(100 if key=='recall' else 1)
    assert all(np.isfinite(v).all()for v in cells.values())
    provenance=dict(source_hashes={str(p):sha(p)for p in paths},figure_code_sha256=sha(__file__),
                    training_groups=list(GROUPS),seeds=list(SEEDS),duration_inputs=list(VIEWS),alpha=.001,
                    primary_ranking_metric='pauc_std_01',readout='base',confidence_intervals=False,
                    scope='Previously consumed real development events. Connected dots are matched checkpoints across input duration; three seeds are not independent new event samples. Synthetic outcomes are not plotted.')
    return cells,provenance
def main():
    cells,provenance=load_audited()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'svg.fonttype':'none',
                         'figure.facecolor':'white','savefig.facecolor':'white'})
    fig,axes=plt.subplots(2,2,figsize=(13.4,10.1))
    fig.subplots_adjust(left=.085,right=.975,bottom=.205,top=.80,wspace=.24,hspace=.34)
    colors=('#2869ad','#bd7800','#258452');jitter=(-.045,0,.045);positions=(-.20,.20)
    configs=[('pauc_std_01','A  Ranking at low false-alarm rates','Standardized partial AUROC (FPR ≤1%)'),
             ('recall','B  Targets found at the frozen thresholds','Recall (%)'),
             ('fa_per10000','C  False alarms at the frozen thresholds','False alarms per 10,000 negatives'),
             ('mcc','D  Classification at the frozen thresholds','Matthews correlation coefficient')]
    for ax,(key,title,ylabel)in zip(axes.flat,configs):
        a=cells[key]
        for gi in range(3):
            for si,(color,shift)in enumerate(zip(colors,jitter)):
                x=np.array(positions)+gi+shift;y=a[gi,:,si]
                ax.plot(x,y,color=color,linewidth=1.25,alpha=.85,zorder=2)
                ax.scatter(x[0],y[0],s=43,marker='o',c=color,edgecolors='white',linewidths=.6,zorder=4)
                ax.scatter(x[1],y[1],s=43,marker='s',c=color,edgecolors='white',linewidths=.6,zorder=4)
            means=a[gi].mean(1);ax.plot(np.array(positions)+gi,means,color='#20252c',linewidth=2.1,marker='D',markersize=5,zorder=5)
        if key=='pauc_std_01':
            ax.set_ylim(min(.45,float(a.min())-.035),1.);ax.axhline(.5,color='#979fa9',linestyle=':',linewidth=1,zorder=0)
            ax.text(2.40,.5,'Chance: 0.5',ha='right',va='bottom',fontsize=8,color='#68727f')
        elif key=='recall':ax.set_ylim(0,100)
        elif key=='mcc':ax.set_ylim(min(-.04,float(a.min())-.05),1.)
        else:
            ax.set_ylim(0,max(40.,float(a.max())*1.18));ax.axhline(10,color='#9c5143',linestyle='--',linewidth=1,zorder=1)
            ax.text(2.40,10,'Nominal target: 10',ha='right',va='bottom',fontsize=8,color='#8c483b')
        ax.set_xlim(-.48,2.48);ax.set_xticks(range(3),LABELS);ax.set_ylabel(ylabel);ax.set_title(title,loc='left',fontweight='bold',pad=10)
        ax.grid(axis='y',color='#e4e7eb',linewidth=.7);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
        ax.spines[['left','bottom']].set_color('#b8bfc8');ax.tick_params(axis='x',length=0,pad=10)
    fig.suptitle('What changes with temporal context and mathematical training?',x=.085,y=.975,ha='left',fontsize=16,fontweight='bold')
    fig.text(.085,.939,'Real-image development data • fixed base descriptors • three matched initializations • 800 updates per trained model',fontsize=10.5,color='#414957')
    handles=[Line2D([0],[0],color=c,linewidth=1.5,label=f'Seed {s}')for c,s in zip(colors,SEEDS)]
    handles.extend([Line2D([0],[0],color='none',marker='o',markerfacecolor='#6b7380',markersize=6,label='1-second image'),
                    Line2D([0],[0],color='none',marker='s',markerfacecolor='#6b7380',markersize=6,label='Four duration images'),
                    Line2D([0],[0],color='#20252c',marker='D',markersize=5,linewidth=2,label='Three-seed mean')])
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.079,.913),ncol=3,frameon=False,columnspacing=2.8,handlelength=2.)
    fig.text(.085,.132,'Each line pairs the same checkpoint: one 1-second view → genuine 0.5, 1, 2 and 4-second views of the same event.',fontsize=10,color='#343d4a')
    fig.text(.085,.104,'One positive reference event per target: one PNG versus four PNGs. All arms share 512 negative FIT examples and separate CAL.',fontsize=9.5,color='#343d4a')
    fig.text(.085,.076,'Metrics average four targets at calibration α = 0.001. Partial AUROC is standardized (chance 0.5); it is not ordinary accuracy.',fontsize=9.5,color='#343d4a')
    fig.text(.085,.048,'Previously consumed DEV, not a fresh confirmatory test. No seed confidence intervals; random controls receive no training.',fontsize=9.5,color='#343d4a')
    fig.text(.085,.022,'False-alarm averages are rescaled observed rates, not a union-of-targets archive rate or deployment guarantee. DINO appears in the report tables.',fontsize=8.8,color='#596271')
    description=json.dumps(provenance,sort_keys=True)
    fig.savefig(R/'comparison.png',dpi=180,metadata={'Title':'HyperNet v46 paired duration and generator comparison','Description':description})
    fig.savefig(R/'comparison.svg',metadata={'Title':'HyperNet v46 paired duration and generator comparison','Description':description,'Creator':'Matplotlib; audited saved real-DEV results only'})
    plt.close(fig);print('Wrote',R/'comparison.png',R/'comparison.svg',flush=True)
if __name__=='__main__':main()
