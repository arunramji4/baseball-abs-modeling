import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
"""Paired five-game-fold release-feature ablations for challenge likelihood."""
import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from train_model import ROOT,features,metrics

out=ROOT/'outputs'/'release-features';out.mkdir(parents=True,exist_ok=True)
d=pd.read_csv(ROOT/'data-train.csv',dtype={'game_id':str})
config=json.loads((ROOT/'outputs/optuna-strike-zone/metrics.json').read_text())
cats=json.loads((ROOT/'outputs/optuna-strike-zone/categories.json').read_text())
base=features(d,cats).drop(columns='catcher_id').join(d[['sz_top','sz_bot']])
membership=pd.read_csv(ROOT/'outputs/optuna-strike-zone/folds.csv',dtype={'game_id':str})
assert d.play_id.equals(membership.play_id)
cols=['release_x','release_z','release_angle_x','release_angle_z','extension']
variants={'baseline':[],**{c:[c] for c in cols},'release_position':['release_x','release_z'],'release_angles':['release_angle_x','release_angle_z'],'all_release':cols}
report={'target':'is_challenge','base_features':list(base.columns),'parameters':{**config['fixed_parameters'],**config['best_parameters']},'missing_rates':d[cols].isna().mean().to_dict(),'variants':{},'limitations':'Same previously used tuning folds; no fresh holdout or per-variant hyperparameter retuning. Release angles describe initial ball trajectory, not a directly measured anatomical arm angle.'}
for label,extra in variants.items():
    x=base.join(d[extra]) if extra else base
    pred=np.zeros(len(d));fold_metrics=[]
    for fold in range(1,6):
        b=np.flatnonzero(membership.fold.to_numpy()==fold);a=np.flatnonzero(membership.fold.to_numpy()!=fold)
        assert not set(d.iloc[a].game_id)&set(d.iloc[b].game_id)
        m=XGBClassifier(**report['parameters']);m.fit(x.iloc[a],d.iloc[a].is_challenge)
        pred[b]=m.predict_proba(x.iloc[b])[:,1]
        result=metrics(d.iloc[b].is_challenge,pred[b]);fold_metrics.append(result)
        print(label,fold,round(result['log_loss'],7),flush=True)
    result={'added_features':extra,'mean_fold_log_loss':float(np.mean([f['log_loss'] for f in fold_metrics])),'pooled_metrics':metrics(d.is_challenge,pred),'fold_metrics':fold_metrics}
    if label!='baseline':
        ref=report['variants']['baseline']
        result['relative_log_loss_improvement_pct']=100*(1-result['mean_fold_log_loss']/ref['mean_fold_log_loss'])
        result['fold_log_loss_changes']=[f['log_loss']-r['log_loss'] for f,r in zip(fold_metrics,ref['fold_metrics'])]
    report['variants'][label]=result
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print('COMPLETE',label,result['mean_fold_log_loss'],result.get('relative_log_loss_improvement_pct',0),flush=True)
pd.DataFrame([{'variant':k,'log_loss':v['mean_fold_log_loss'],'auc':v['pooled_metrics']['roc_auc'],'improvement_pct':v.get('relative_log_loss_improvement_pct',0),'folds_improved':sum(c<0 for c in v.get('fold_log_loss_changes',[]))} for k,v in report['variants'].items()]).to_csv(out/'comparison.csv',index=False)
