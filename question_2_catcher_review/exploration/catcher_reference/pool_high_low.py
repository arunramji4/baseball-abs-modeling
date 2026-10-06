from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/mariners-matplotlib')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(__file__).resolve().parent
x=pd.read_csv(out/'catcher_pitches_excluding_game.csv')
groups=['All pitches','Fastballs','Breaking','Offspeed'];rows=[]
for group in groups:
    s=x if group=='All pitches' else x[x.pitch_group.eq(group)]
    for level in ['High','Low']:
        q=s[s.vertical.eq(level)];n=len(q);b=int(q.pitch_result.eq('ball').sum());c=int(q.defensive_challenge.sum());w=int(q.defensive_success.sum())
        rows.append(dict(pitch_group=group,location=level,called_pitches=n,called_balls=b,challenges=c,overturns=w,success_rate=w/c if c else np.nan,challenge_rate=c/b if b else np.nan))
r=pd.DataFrame(rows);r.to_csv(out/'high_low_pooled_counts.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(12,6))
for ax,metric,title in zip(axes,['success_rate','challenge_rate'],['Challenge success','Challenge frequency on balls']):
    values=np.array([[r[(r.pitch_group==g)&(r.location==l)][metric].iloc[0] for g in groups] for l in ['High','Low']])
    ax.imshow(np.ma.masked_invalid(values),cmap='Blues',vmin=0,vmax=1 if metric=='success_rate' else max(.05,r.challenge_rate.max()),aspect='auto')
    for yi,l in enumerate(['High','Low']):
        for xi,g in enumerate(groups):
            a=r[(r.pitch_group==g)&(r.location==l)].iloc[0]
            pct='N/A' if pd.isna(a[metric]) else f'{a[metric]:.0%}'
            text=f'{pct}\n{a.overturns}/{a.challenges} wins\n{a.called_pitches} calls' if metric=='success_rate' else f'{pct}\n{a.challenges}/{a.called_balls}\nballs challenged'
            vmax=1 if metric=='success_rate' else max(.05,r.challenge_rate.max())
            color='white' if np.isfinite(a[metric]) and a[metric]>vmax*.55 else 'black'
            ax.text(xi,yi,text,ha='center',va='center',fontsize=10,color=color)
    ax.set_xticks(range(4),groups,rotation=20);ax.set_yticks([0,1],['High','Low']);ax.set_title(title)
fig.suptitle('Catcher EXAMPLE_CATCHER — high/low, both batter hands pooled\nGame 00000000 excluded',fontsize=14)
fig.text(.5,.02,'High/low split at each batter’s personal zone midpoint. Pitch families are estimates.\nDefensive outcomes do not identify whether the catcher or pitcher initiated the challenge.',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.09,1,.88]);fig.savefig(out/'high_low_pooled_heatmap.png',dpi=160)
print(r.to_string(index=False))
