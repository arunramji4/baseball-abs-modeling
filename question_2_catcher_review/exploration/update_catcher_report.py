from pathlib import Path
import sys
import json
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[2] / 'question_1_modeling'))
from experiments.evaluate_batting_order import infer_order

here=Path(__file__).resolve().parent
g=pd.read_csv(here/'game_00000000.csv',dtype={'game_id':str})
g['batting_order']=infer_order(g)[0]
known=g.dropna(subset=['batting_order']).groupby(['is_top_inning','batter_id']).batting_order.agg(lambda s:s.iloc[0] if s.nunique()==1 else np.nan)
g['batting_order']=g.apply(lambda r:r.batting_order if pd.notna(r.batting_order) else known.get((r.is_top_inning,r.batter_id),np.nan),axis=1)
# A complete nine-batter roster with eight stable slots resolves the last slot.
for half, side in g.groupby('is_top_inning'):
    batters=side.batter_id.unique()
    assignments=side.dropna(subset=['batting_order']).groupby('batter_id').batting_order.agg(lambda values: values.iloc[0] if values.nunique()==1 else np.nan).dropna()
    missing=[b for b in batters if b not in assignments.index]
    unused=set(range(1,10))-set(assignments.astype(int))
    if len(batters)==9 and len(assignments)==8 and len(missing)==1 and len(unused)==1:
        g.loc[g.is_top_inning.eq(half)&g.batter_id.eq(missing[0]),'batting_order']=unused.pop()
x=g[g.is_top_inning.eq(True)&g.pitch_result.eq('ball')].copy()
radius=1.45
hx=x.plate_x.abs()*12-8.5
vz=np.maximum((x.sz_bot-x.plate_z)*12,(x.plate_z-x.sz_top)*12)
x['horizontal_margin_in']=hx-radius
x['vertical_margin_in']=vz-radius
# True nearest Euclidean separation handles pitches near a corner.
inside=np.minimum(-hx,-vz)
x['contact_margin_in']=np.where(inside.ge(0),-inside,np.sqrt(hx.clip(lower=0)**2+vz.clip(lower=0)**2))-radius
x=x[x.contact_margin_in.abs().le(2)|x.is_challenge.eq(1)].copy()
settings=json.loads((here.parents[1]/'question_1_modeling/outputs/optuna-success-hands/metrics.json').read_text())
training=pd.read_csv(here.parents[1]/'question_1_modeling/data-train.csv',dtype={'game_id':str})
training=training[training.is_challenge.eq(1)&~training.game_id.str.zfill(8).eq('00000000')]
from model_features import prepare as prepare_features
def prepare(d):
    return prepare_features(d,settings['features'])
model=XGBClassifier(**settings['fixed_parameters'],**settings['best_parameters'])
model.fit(prepare(training),training.is_success.astype(int))
x['p_overturn']=model.predict_proba(prepare(x))[:,1]
x['p_upheld']=1-x.p_overturn
x.to_csv(here/'game_00000000_close_challengeable_balls.csv',index=False)
print('Updated close pitches:',len(x))
