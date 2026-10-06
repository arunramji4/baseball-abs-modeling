import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
"""Paired five-fold comparison adding personal strike-zone boundaries."""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier
from train_model import ROOT, features, CATEGORICAL, SEED, metrics

OUT=ROOT/'outputs'/'xgboost-strike-zone'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    settings=json.loads((ROOT/'outputs'/'optuna'/'metrics.json').read_text())
    params={**settings['fixed_parameters'],**settings['best_parameters']}
    d=pd.read_csv(ROOT/'data-train.csv',dtype={'game_id':str})
    t=pd.read_csv(ROOT/'data-test.csv',dtype={'game_id':str})
    cats={c:sorted(d[c].fillna('missing').astype(str).unique()) for c in CATEGORICAL}
    base=features(d,cats).drop(columns='catcher_id')
    extended=base.join(d[['sz_top','sz_bot']])
    folds=list(GroupKFold(n_splits=5,shuffle=True,random_state=SEED).split(d,groups=d.game_id))
    report={'features':list(extended.columns),'parameters':params,'comparisons':{},
            'method':'Same five game folds and frozen Optuna-selected parameters; no retuning or early stopping.',
            'limitation':'Development CV on data previously used for feature and parameter selection.'}
    all_predictions={}
    for label,x in [('tuned_baseline',base),('add_sz_top_sz_bot',extended)]:
        p=np.zeros(len(d));scores=[]
        for fold,(a,b) in enumerate(folds,1):
            assert not set(d.iloc[a].game_id)&set(d.iloc[b].game_id)
            m=XGBClassifier(**params);m.fit(x.iloc[a],d.iloc[a].is_challenge)
            p[b]=m.predict_proba(x.iloc[b])[:,1]
            scores.append({'fold':fold,**metrics(d.iloc[b].is_challenge,p[b])})
            print(label,fold,scores[-1]['log_loss'],flush=True)
        all_predictions[label]=p
        report['comparisons'][label]={'mean_fold_log_loss':float(np.mean([s['log_loss'] for s in scores])),
                                     'pooled_metrics':metrics(d.is_challenge,p),'fold_metrics':scores}
    before=report['comparisons']['tuned_baseline'];after=report['comparisons']['add_sz_top_sz_bot']
    report['relative_improvement_pct']=100*(1-after['mean_fold_log_loss']/before['mean_fold_log_loss'])
    report['folds_improved']=sum(y['log_loss']<x['log_loss'] for x,y in zip(before['fold_metrics'],after['fold_metrics']))
    final=XGBClassifier(**params);final.fit(extended,d.is_challenge)
    final.save_model(OUT/'is_challenge.ubj')
    xt=features(t,cats).drop(columns='catcher_id').join(t[['sz_top','sz_bot']])
    t['p_challenge']=final.predict_proba(xt)[:,1]
    t['challenge_source']=np.where(t.pitch_result.eq('ball'),'pitching_team','hitting_team')
    assert t.p_challenge.between(0,1).all()
    t.to_csv(OUT/'data-test-predictions.csv',index=False)
    pd.DataFrame({'play_id':d.play_id,**all_predictions}).to_csv(OUT/'out-of-fold-predictions.csv',index=False)
    pd.DataFrame({'feature':extended.columns,'importance':final.feature_importances_}).to_csv(OUT/'feature-importance.csv',index=False)
    (OUT/'categories.json').write_text(json.dumps(cats,indent=2)+'\n')
    (OUT/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print('improvement_pct',report['relative_improvement_pct'],'folds_improved',report['folds_improved'],flush=True)


if __name__=='__main__':main()
