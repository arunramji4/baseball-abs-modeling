import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
"""Pitch-feature ablations for success conditional on an observed challenge."""
import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from train_model import ROOT,metrics
from train_challenge_success import prepare

out=ROOT/'outputs'/'success-pitch-features';out.mkdir(parents=True,exist_ok=True)
d=pd.read_csv(ROOT/'data-train.csv',dtype={'game_id':str})
d=d[d.is_challenge.eq(1)].reset_index(drop=True);y=d.is_success.astype(int)
base=prepare(d)
config=json.loads((ROOT/'outputs/optuna-success/metrics.json').read_text())
params={**config['fixed_parameters'],**config['best_parameters']}
member=pd.read_csv(ROOT/'outputs/optuna-success/splits.csv',dtype={'game_id':str})
assert d.play_id.equals(member.play_id)
a=np.flatnonzero(member.partition.eq('tuning'));b=np.flatnonzero(member.partition.eq('holdout'))
splits=[]
for seed in config['seeds']:
    f=member[f'fold_seed_{seed}'].to_numpy()
    for fold in range(1,6):
        train=np.flatnonzero((f>0)&(f!=fold));val=np.flatnonzero(f==fold)
        assert not set(d.iloc[train].game_id)&set(d.iloc[val].game_id)
        splits.append((train,val))
individual=['batter_hand','pitcher_hand','velocity','break_x','break_z','release_x','release_z','release_angle_x','release_angle_z','extension']
variants={'baseline':[],**{c:[c] for c in individual},'both_hands':['batter_hand','pitcher_hand'],'both_breaks':['break_x','break_z'],'velocity_and_breaks':['velocity','break_x','break_z'],'release_positions':['release_x','release_z'],'release_angles':['release_angle_x','release_angle_z'],'all_release':['release_x','release_z','release_angle_x','release_angle_z','extension'],'hands_velocity_breaks':['batter_hand','pitcher_hand','velocity','break_x','break_z'],'all_pitch_features':individual}
def frame(extra):
    x=base.join(d[extra]) if extra else base.copy()
    for c in ['batter_hand','pitcher_hand']:
        if c in extra:x[c]=pd.Categorical(x[c].fillna('missing'),categories=['Left','Right','Switch','missing'])
    return x
report={'target':'is_success among challenged pitches','base_features':list(base.columns),'parameters':params,'variants':{},'validation':'Original three seeds x five grouped folds on tuning games; original holdout evaluated only for baseline and CV-selected best addition. Fixed hyperparameters, no retuning.'}
for label,extra in variants.items():
    x=frame(extra);folds=[]
    for train,val in splits:
        m=XGBClassifier(**params);m.fit(x.iloc[train],y.iloc[train]);p=m.predict_proba(x.iloc[val])[:,1]
        folds.append({**metrics(y.iloc[val],p),'accuracy':float(accuracy_score(y.iloc[val],p>=.5))})
    loss=float(np.mean([f['log_loss'] for f in folds]))
    result={'added_features':extra,'mean_log_loss':loss,'mean_accuracy':float(np.mean([f['accuracy'] for f in folds])),'fold_metrics':folds}
    if label!='baseline':
        ref=report['variants']['baseline'];changes=[f['log_loss']-r['log_loss'] for f,r in zip(folds,ref['fold_metrics'])]
        result.update(relative_improvement_pct=100*(1-loss/ref['mean_log_loss']),folds_improved=sum(v<0 for v in changes),fold_changes=changes)
    report['variants'][label]=result
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(label,loss,result.get('relative_improvement_pct',0),result.get('folds_improved',0),flush=True)
best=min(report['variants'],key=lambda k:report['variants'][k]['mean_log_loss'])
report['cv_selected_variant']=best;report['holdout']={}
for label in dict.fromkeys(['baseline',best]):
    x=frame(variants[label]);m=XGBClassifier(**params);m.fit(x.iloc[a],y.iloc[a]);p=m.predict_proba(x.iloc[b])[:,1]
    report['holdout'][label]={**metrics(y.iloc[b],p),'accuracy':float(accuracy_score(y.iloc[b],p>=.5))}
report['limitations']='CV has been used for previous tuning and many feature comparisons. Existing holdout previously evaluated, so not an untouched external test. Results measure prediction conditional on a challenge, not causal effects.'
(out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
pd.DataFrame([{'variant':k,'cv_log_loss':v['mean_log_loss'],'improvement_pct':v.get('relative_improvement_pct',0),'folds_improved':v.get('folds_improved',0),'cv_accuracy':v['mean_accuracy']} for k,v in report['variants'].items()]).to_csv(out/'comparison.csv',index=False)
print('HOLDOUT',json.dumps(report['holdout']),flush=True)
