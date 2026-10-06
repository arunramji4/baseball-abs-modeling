from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/mariners-matplotlib')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Ellipse
from matplotlib.lines import Line2D
from pitch_clusters import classify
here=Path(__file__).resolve().parent
report_path=here/'game_00000000_part2.md'
existing_intro=report_path.read_text().split('## 1.',1)[0] if report_path.exists() else None
x=pd.read_csv(here/'game_00000000_close_challengeable_balls.csv').sort_values(['pa_of_game','pitch_of_pa']).reset_index(drop=True)
x['pitch_number']=np.arange(1,len(x)+1)
x['pitch_group']=classify(x)
zone_height=float((x.sz_top-x.sz_bot).mean())
x['normalized_height']=(x.plate_z-x.sz_bot)/(x.sz_top-x.sz_bot)*zone_height
game=pd.read_csv(here/'game_00000000.csv').sort_values(['pa_of_game','pitch_of_pa'])
balance=2
remaining_by_play={}
for r in game.itertuples():
    remaining_by_play[r.play_id]=balance
    home_actual=(r.challenge_source=='pitching_team' and bool(r.is_top_inning)) or (r.challenge_source=='hitting_team' and not bool(r.is_top_inning))
    if r.is_challenge==1 and home_actual and r.is_success==0:
        balance-=1
x['challenges_remaining']=x.play_id.map(remaining_by_play)
def summary_table(likely):
    selected=x[x.p_overturn.ge(.5).eq(likely)]
    header='| Pitch # | Overturn likelihood | Pitch cluster | Inning | Score (+/−) | Challenges remaining | MOB | WP change if correct | Catcher movement | Hitter |\n|---:|---:|---|---|---:|---:|---|---|---|---|'
    rows=[f'| {r.pitch_number} | {r.p_overturn:.1%} | {r.pitch_group} | Top {r.inning} | {r.pre_score_home-r.pre_score_away:+d} | {r.challenges_remaining} | ? | ? | ? | ? |' for r in selected.itertuples()]
    return header+'\n'+'\n'.join(rows)
paths=[]
for likely,name,title in [(True,'game_00000000_likely_overturned.png','1. Likely overturned'),(False,'game_00000000_unlikely_overturned.png','2. Unlikely overturned')]:
    s=x[x.p_overturn.ge(.5).eq(likely)]
    fig,ax=plt.subplots(figsize=(8,6),facecolor='#eef2f6')
    fig.subplots_adjust(left=.04,right=.62,bottom=.06,top=.90)
    ax.set_facecolor('#eef2f6')
    ax.add_patch(Rectangle((-17/24,0),17/12,zone_height,facecolor='#ffffff',edgecolor='#515d6a',linewidth=1.2))
    colors=np.where(s.is_challenge.eq(1),np.where(s.is_success.eq(1),'#23954b','#d43c3c'),'#d8dadd')
    family_colors={'Fastballs':'#92a7c1','Breaking':'#c7b69a','Offspeed':'#a9b8ad','Unknown':'#cccccc'}
    for r in s.itertuples():
        color=family_colors[r.pitch_group]
        edge=('#15803d' if r.is_success==1 else '#dc2626') if r.is_challenge==1 else '#56616d'
        radius=1.45/12
        ax.add_patch(Ellipse((r.plate_x,r.normalized_height),2*radius,2*radius,facecolor=color,edgecolor=edge,alpha=1,linewidth=2.7 if r.is_challenge==1 else .9,zorder=3))
        near=s[(s.plate_x-r.plate_x).abs().lt(.11)&(s.normalized_height-r.normalized_height).abs().lt(.12)]
        offset=(12 if r.pitch_number%2 else -12,12 if r.pitch_number%2 else -12) if len(near)>1 else (0,0)
        if r.pitch_number==6:offset=(-30,-16)
        if r.pitch_number==8:offset=(28,-24)
        ax.annotate(str(r.pitch_number),(r.plate_x,r.normalized_height),xytext=offset,textcoords='offset points',ha='center',va='center',fontsize=8,fontweight='bold',color='black',arrowprops=dict(arrowstyle='-',color='#777777',linewidth=.6) if offset!=(0,0) else None,zorder=4)
    ax.set_xlim(-1.55,1.55);ax.set_ylim(-.45,zone_height+.5)
    ax.set_aspect('equal',adjustable='box')
    ax.set_yticks([]);ax.set_xticks([])
    for spine in ax.spines.values():spine.set_visible(False)
    if likely:
        family_legend=ax.legend(handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=family_colors[family],markeredgecolor='#56616d',markersize=9,label=family) for family in ['Fastballs','Breaking','Offspeed']],loc='upper left',bbox_to_anchor=(.88,.94),frameon=True,facecolor='white',edgecolor='#d5dce3',fontsize=8,title='Pitch family',title_fontsize=8,borderpad=.5,labelspacing=.4,handletextpad=.4)
        ax.add_artist(family_legend)
        ax.legend(handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor='white',markeredgecolor=edge,markeredgewidth=2,markersize=9,label=label) for edge,label in [('#15803d','Overturned'),('#dc2626','Upheld'),('#56616d','Not challenged')]],loc='upper left',bbox_to_anchor=(.88,.55),frameon=True,facecolor='white',edgecolor='#d5dce3',fontsize=8,title='Challenge outcome',title_fontsize=8,borderpad=.5,labelspacing=.4,handletextpad=.4)
    ax.grid(False)
    fig.savefig(here/name,dpi=180,bbox_inches='tight');plt.close(fig);paths.append(name)
rows=[]
for r in x.itertuples():
    observed=('Overturned' if r.is_success==1 else 'Upheld') if r.is_challenge==1 else 'Not challenged'
    slot=str(int(r.batting_order)) if pd.notna(r.batting_order) else '?'
    rows.append(f'| **{r.pitch_number}** | T{r.inning} | {slot} | {r.pre_score_home-r.pre_score_away:+d} | {r.balls}–{r.strikes} | {r.outs} | {r.horizontal_margin_in:+.2f} | {r.vertical_margin_in:+.2f} | {r.contact_margin_in:+.2f} | {r.p_overturn:.1%} / {r.p_upheld:.1%} | {observed} |')
report='''# Part 2 — Home catcher ABS review, game 00000000

Close, challengeable balls; use the selected game and catcher.

## 1. Likely overturned

![Likely overturned close pitches](game_00000000_likely_overturned.png)

## 2. Unlikely overturned

![Unlikely overturned close pitches](game_00000000_unlikely_overturned.png)

## Numbered pitch reference

Score is the catcher’s team lead (+), deficit (−), or tie (+0), before the pitch. Count and outs are also pre-pitch. Batter # is the reconstructed lineup slot.

Distances below are in inches: positive = clearance outside the zone, negative = overlap. Horizontal and vertical margins include the approximate ball radius; contact margin combines them with corner geometry. Pitch fill colors identify estimated pitch family. Green outlines mark successful challenges, red outlines failed challenges, and gray outlines unchallenged pitches.

| Pitch # | Inning | Batter # | Score +/- | Count | Outs | Horizontal margin | Vertical margin | Contact margin | Overturn / upheld | Actual result |
|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---|
'''+ '\n'.join(rows)+'''

## Notes by pitch

'''+ '\n\n'.join(f'### Pitch {r.pitch_number}\n\nBatter **#{int(r.batting_order)}** (`{r.batter_id}`), pitcher `{r.pitcher_id}`. Top {r.inning}; catcher team score **{r.pre_score_home-r.pre_score_away:+d}**, count **{r.balls}–{r.strikes}**, **{r.outs} outs**.\n\nLocation: X **{r.plate_x:.3f} ft**, Z **{r.plate_z:.3f} ft**; zone bottom **{r.sz_bot:.3f} ft**, top **{r.sz_top:.3f} ft**. Horizontal margin **{r.horizontal_margin_in:+.2f} in**, vertical margin **{r.vertical_margin_in:+.2f} in**, contact margin **{r.contact_margin_in:+.2f} in**.\n\nPitch family **{r.pitch_group}** (estimated). Velocity **{r.velocity:.1f} mph**; horizontal movement **{r.break_x:+.1f} in**, vertical movement **{r.break_z:+.1f} in**. Model: **{r.p_overturn:.1%} overturn / {r.p_upheld:.1%} upheld**. Actual: **{("overturned" if r.is_success==1 else "upheld") if r.is_challenge==1 else "not challenged"}**.\n\n**Your notes:** ' for r in x.itertuples())+'''

**Method:** Likely = at least 50% overturn probability; unlikely = below 50%. Probabilities come from the tuned location/zone/handedness success model refitted excluding this game’s outcomes. Its hyperparameters were selected previously using the wider dataset. Charts use a plain strike-zone box and numbered center markers. The surrounding circle shows an approximate 1.45-inch ball radius at equal horizontal/vertical scale. Vertical centers are adjusted to a common mean zone height, so the display is an approximate normalized view, not raw tracking geometry. A center outside the box can still have ball/zone overlap. All pitches are circles. Muted blue identifies fastballs, tan breaking, and sage offspeed; outline color identifies actual challenge outcome. Offspeed is an inferred arm-side, lower-velocity cluster, not a verified changeup/splitter label, and can include sinkers. Fill identifies pitch family; outline identifies actual challenge outcomes. Pitch families use the saved three-cluster model fitted on other games from velocity and handedness-normalized horizontal/vertical movement; labels are approximate and may mix sinkers with offspeed. Heights are normalized to each batter’s zone so different zone heights are comparable. The close-pitch screen uses an approximate two-inch ball-contact margin and includes observed actual challenges. All shown pitches were originally called balls. Model probabilities are not official ABS rulings for unchallenged pitches. Source data contains called pitches only.
'''
details=report.split('## Numbered pitch reference',1)[1]
(here/'game_00000000_pitch_details.md').write_text('# Pitch details and notes — game 00000000\n\n## Numbered pitch reference'+details)
visual=report.split('## Numbered pitch reference',1)[0]
for likely,image_name in [(True,'game_00000000_likely_overturned.png'),(False,'game_00000000_unlikely_overturned.png')]:
    marker=f']({image_name})'
    visual=visual.replace(marker,marker+'\n\n'+summary_table(likely))
visual=visual.replace('## 1. Likely overturned','Score is the catcher’s team lead (+) or deficit (−) before the pitch. Challenges remaining is the home team’s balance before a challenge: two initially, with unsuccessful challenges consuming one and successful challenges retaining it. MOB and WP are left as requested placeholders.\n\n## 1. Likely overturned')
if existing_intro is not None:
    visual=existing_intro+'## 1.'+visual.split('## 1.',1)[1]
(here/'game_00000000_part2.md').write_text(visual.rstrip()+'\n')
x.to_csv(here/'game_00000000_numbered_close_pitches.csv',index=False)
assert x.pitch_number.is_unique
print('Likely:',int(x.p_overturn.ge(.5).sum()),'Unlikely:',int(x.p_overturn.lt(.5).sum()))
