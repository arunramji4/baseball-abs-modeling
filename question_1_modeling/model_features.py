"""Pre-decision features shared by the final training and prediction code."""
import numpy as np
import pandas as pd

CATEGORIES={'batter_hand':['Left','Right','Switch','missing'],'pitcher_hand':['Left','Right','Switch','missing'],'pitch_result':['ball','called_strike','missing']}
def prepare(data,names):
    d=data.copy()
    top=d.is_top_inning.astype(str).str.lower().eq('true')
    if not d.pitch_result.isin(['ball','called_strike']).all():
        raise ValueError('Unexpected original pitch call')
    home=(d.pitch_result.eq('ball')&top)|(d.pitch_result.eq('called_strike')&~top)
    d['challenger_score_diff']=np.where(home,d.pre_score_home-d.pre_score_away,d.pre_score_away-d.pre_score_home)
    x=d[names].copy()
    for c in names:
        if c in CATEGORIES:x[c]=pd.Categorical(x[c].fillna('missing'),categories=CATEGORIES[c])
        else:x[c]=pd.to_numeric(x[c],errors='raise')
    return x
