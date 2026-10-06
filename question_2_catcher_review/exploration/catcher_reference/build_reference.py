from pathlib import Path
import os
os.environ.setdefault('LOKY_MAX_CPU_COUNT','4')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
x=pd.read_csv(ROOT.parent/'question_1_modeling/data-train.csv',dtype={'game_id':str})
x=x[~x.game_id.str.zfill(8).eq('00000000')].dropna(subset=['velocity','break_x','break_z']).copy()
x['arm_side_break']=x.break_x*np.where(x.pitcher_hand.eq('Right'),1,-1)
cols=['velocity','arm_side_break','break_z']
s=StandardScaler();m=KMeans(n_clusters=3,random_state=42,n_init=10)
x['cluster']=m.fit_predict(s.fit_transform(x[cols]))
centers=x.groupby('cluster')[cols].mean()
fast=centers.velocity.idxmax();breaking=centers.drop(index=fast).arm_side_break.idxmin();off=next(i for i in centers.index if i not in [fast,breaking])
labels={fast:'Fastballs',breaking:'Breaking',off:'Offspeed'}
x['pitch_group']=x.cluster.map(labels)
centers['pitch_group']=centers.index.map(labels);centers.to_csv(OUT/'cluster_centers.csv')
x=x[x.catcher_id.eq('EXAMPLE_CATCHER')].copy()
x['vertical']=np.where(x.plate_z.ge((x.sz_top+x.sz_bot)/2),'High','Low')
x['horizontal']=np.where(x.plate_x.lt(0),'Left','Right')
x['defensive_challenge']=x.is_challenge.eq(1)&x.challenge_source.eq('pitching_team')
x['defensive_success']=x.defensive_challenge&x.is_success.eq(1)
records=[];groups=['All pitches','Fastballs','Breaking','Offspeed']
for group in groups:
    sub=x if group=='All pitches' else x[x.pitch_group.eq(group)]
    for hand in ['Left','Right']:
        for vertical in ['High','Low']:
            for horizontal in ['Left','Right']:
                p=sub[sub.batter_hand.eq(hand)&sub.vertical.eq(vertical)&sub.horizontal.eq(horizontal)]
                n=len(p);balls=int(p.pitch_result.eq('ball').sum());ch=int(p.defensive_challenge.sum());win=int(p.defensive_success.sum())
                records.append(dict(pitch_group=group,batter_hand=hand,vertical=vertical,horizontal=horizontal,called_pitches=n,called_balls=balls,challenges=ch,overturns=win,success_rate=win/ch if ch else np.nan,challenge_rate=ch/balls if balls else np.nan))
r=pd.DataFrame(records);r.to_csv(OUT/'heatmap_counts.csv',index=False)
x.to_csv(OUT/'catcher_pitches_excluding_game.csv',index=False)
for metric,title,name in [('success_rate','Challenge success','challenge_success_heatmaps.png'),('challenge_rate','Challenge frequency on called balls','challenge_frequency_heatmaps.png')]:
    fig,axes=plt.subplots(4,2,figsize=(10,14))
    vmax=1 if metric=='success_rate' else max(.05,r[metric].max())
    for i,group in enumerate(groups):
        for j,hand in enumerate(['Left','Right']):
            ax=axes[i,j];sub=r[r.pitch_group.eq(group)&r.batter_hand.eq(hand)];vals=np.full((2,2),np.nan)
            for yi,v in enumerate(['High','Low']):
                for xi,h in enumerate(['Left','Right']):
                    row=sub[sub.vertical.eq(v)&sub.horizontal.eq(h)].iloc[0];vals[yi,xi]=row[metric]
            ax.imshow(np.ma.masked_invalid(vals),cmap='Blues',vmin=0,vmax=vmax)
            for yi,v in enumerate(['High','Low']):
                for xi,h in enumerate(['Left','Right']):
                    row=sub[sub.vertical.eq(v)&sub.horizontal.eq(h)].iloc[0]
                    pct='No challenges' if pd.isna(row[metric]) else f'{row[metric]:.0%}'
                    text=(f'{pct}\n{row.overturns}/{row.challenges} overturned\n{row.called_pitches} calls' if metric=='success_rate' else f'{pct}\n{row.challenges}/{row.called_balls} balls challenged')
                    color='white' if np.isfinite(vals[yi,xi]) and vals[yi,xi]>vmax*.55 else 'black'
                    ax.text(xi,yi,text,ha='center',va='center',fontsize=9,color=color)
            ax.set_xticks([0,1],['Left (x < 0)','Right (x >= 0)']);ax.set_yticks([0,1],['High','Low']);ax.set_title(f'{group} · {hand.lower()}-handed batter')
    fig.suptitle(f'Catcher reference — {title}',fontsize=13)
    fig.text(.5,.012,'High/low split at personalized zone midpoint. Left/right = plate_x sign, not inside/outside.\nPitch families are clustering estimates. Sparse cells are not reliable measures of catcher skill.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.045,1,.94]);fig.savefig(OUT/name,dpi=150);plt.close(fig)
(OUT/'REFERENCE.txt').write_text('Generic catcher reference; counts are computed from local private inputs.')
print(centers.to_string());print('Saved heatmaps and checked sample totals.')
