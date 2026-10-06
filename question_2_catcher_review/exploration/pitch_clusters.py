"""Reusable broad pitch families inferred without game 00000000 or outcomes."""
from pathlib import Path
import os
os.environ.setdefault('LOKY_MAX_CPU_COUNT','4')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
ROOT=Path(__file__).resolve().parent.parent
MODEL=ROOT/'exploration/catcher_reference/pitch_clusters.joblib'
COLS=['velocity','arm_side_break','break_z']
def inputs(d):
    return d.assign(arm_side_break=d.break_x*np.where(d.pitcher_hand.eq('Right'),1,-1))[COLS]
def load_or_fit():
    if MODEL.exists():return joblib.load(MODEL)
    d=pd.read_csv(ROOT.parent/'question_1_modeling/data-train.csv',dtype={'game_id':str})
    d=d[~d.game_id.str.zfill(8).eq('00000000')].dropna(subset=['velocity','break_x','break_z'])
    scaler=StandardScaler();km=KMeans(n_clusters=3,random_state=42,n_init=10)
    labels=km.fit_predict(scaler.fit_transform(inputs(d)))
    centers=inputs(d).assign(cluster=labels).groupby('cluster')[COLS].mean()
    fast=centers.velocity.idxmax();breaking=centers.drop(index=fast).arm_side_break.idxmin()
    off=next(i for i in centers.index if i not in [fast,breaking])
    result={'scaler':scaler,'model':km,'labels':{fast:'Fastballs',breaking:'Breaking',off:'Offspeed'},'columns':COLS}
    MODEL.parent.mkdir(exist_ok=True);joblib.dump(result,MODEL)
    return result
def classify(d):
    m=load_or_fit();values=inputs(d);valid=values.notna().all(axis=1)
    out=pd.Series('Unknown',index=d.index)
    out.loc[valid]=[m['labels'][i] for i in m['model'].predict(m['scaler'].transform(values.loc[valid]))]
    return out
