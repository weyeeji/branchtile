"""Show natural costs together with frozen quality/length outcomes."""
import hashlib
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'results/native_precision_natural_development_v1'
source=folder/'analysis.json';data=json.loads(source.read_text())
assert data['status']=='completed'
fig,axes=plt.subplots(1,2,figsize=(9.2,4.8),sharey=True)
for ax,prefix,title in zip(axes,('original','new'),('Development report A','Development report B')):
    rows=[next(row for row in data['rows'] if row['arm']==prefix+'_'+kind) for kind in ('bf16','fp8')]
    for i,row in enumerate(rows):
        assert row['cold_AA']['all_equal']
        anchor=statistics.median(run['anchor_seconds'] for run in row['runs'])
        gen=row['median_same_trajectory_generate_seconds']
        ax.bar(i,anchor,.52,color='#91b6ce',label='Cold q0 anchor' if i==0 else None)
        ax.bar(i,gen,.52,bottom=anchor,color='#256384',label='Natural question group' if i==0 else None)
        ax.text(i,anchor+gen+.9,f'{anchor+gen:.2f} s',ha='center',fontsize=11,fontweight='bold')
        run=row['runs'][0]
        label=f"{run['output_tokens']:,} tokens\nstrict {round(run['exact']*3)}/3; rounded {round(run['rounded']*3)}/3"
        ax.text(i,-5,label,ha='center',va='top',fontsize=9,clip_on=False)
    ax.set_xticks([0,1],['BF16 KV','FP8 KV (unit scale)'])
    ax.set_title(title);ax.set_ylim(0,48);ax.set_xlim(-.6,1.6)
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    ax.spines[['right','top']].set_visible(False)
axes[0].set_ylabel('Anchor + generation wall time (seconds)')
axes[0].legend(loc='upper left',fontsize=8,frameon=False)
fig.suptitle('Native full-history precision controls: cost and quality must be read together',fontsize=11,y=.97)
fig.text(.5,.045,'Median of two identical cold A/A trajectories per arm; 2 reports / 6 questions / 1 seed.\nDifferent precision, output lengths and GPU pairs: descriptive pilot, no quality noninferiority or causal speedup claim.',ha='center',fontsize=8)
fig.subplots_adjust(bottom=.24,top=.88,wspace=.15)
for suffix in ('png','pdf'):fig.savefig(folder/f'quality_cost.{suffix}',dpi=180,bbox_inches='tight')
(folder/'figure_receipt.json').write_text(json.dumps({'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'plot_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'median_rule':'Two A/A runs only after exact token/APC/scheduled-step equality.',
    'quality_denominator':'Three questions per report; repeats are not additional quality samples.',
    'files':{f'quality_cost.{s}':hashlib.sha256((folder/f'quality_cost.{s}').read_bytes()).hexdigest() for s in ('png','pdf')}},indent=2)+'\n')
