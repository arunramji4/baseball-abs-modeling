"""Fit final selected models and create the Q1 CSV; optionally reproduce validation."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold,GroupShuffleSplit
from sklearn.metrics import log_loss,roc_auc_score,accuracy_score
from xgboost import XGBClassifier
from model_features import prepare
ROOT=Path(__file__).resolve().parent

def scores(y,p):
    return {'log_loss':float(log_loss(y,p)),'roc_auc':float(roc_auc_score(y,p)),'accuracy':float(accuracy_score(y,p>=.5))}
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluate',action='store_true',help='Reproduce development CV and success holdout before full refitting')
    args=parser.parse_args()
    config=json.loads((ROOT/'final_config.json').read_text())
    d=pd.read_csv(ROOT/'data-train.csv',dtype={'game_id':str})
    test=pd.read_csv(ROOT/'data-test.csv',dtype={'game_id':str})
    out=ROOT/'outputs/final';out.mkdir(exist_ok=True)
    report={}
    for task,target in [('challenge','is_challenge'),('success','is_success')]:
        rows=d if task=='challenge' else d[d.is_challenge.eq(1)].reset_index(drop=True)
        y=rows[target].astype(int);spec=config[task];x=prepare(rows,spec['features'])
        params={**spec['fixed_parameters'],**spec['best_parameters']}
        if args.evaluate:
            if task=='challenge':
                splits=list(GroupKFold(5,shuffle=True,random_state=42).split(rows,groups=rows.game_id))
            else:
                a,b=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=8675309).split(rows,groups=rows.game_id))
                splits=[(a[i],a[j]) for seed in [42,123,2026] for i,j in GroupKFold(5,shuffle=True,random_state=seed).split(rows.iloc[a],groups=rows.iloc[a].game_id)]
            results=[]
            for i,(train,val) in enumerate(splits,1):
                assert not set(rows.iloc[train].game_id)&set(rows.iloc[val].game_id)
                m=XGBClassifier(**params);m.fit(x.iloc[train],y.iloc[train]);results.append(scores(y.iloc[val],m.predict_proba(x.iloc[val])[:,1]))
                print(task,'fold',i,results[-1]['log_loss'],flush=True)
            report[task]={'fold_metrics':results,'mean_cv_log_loss':float(np.mean([v['log_loss'] for v in results]))}
            if task=='success':
                m=XGBClassifier(**params);m.fit(x.iloc[a],y.iloc[a]);report[task]['holdout']=scores(y.iloc[b],m.predict_proba(x.iloc[b])[:,1])
        final=XGBClassifier(**params);final.fit(x,y);final.save_model(out/f'{target}.ubj')
        print('Saved',target,flush=True)
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    if args.evaluate:
        report['limitation']='Development validation; folds and existing success holdout have been used for prior selection. No fresh external test.'
        (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    from build_submission import main as build
    build(model_dir=out)
if __name__=='__main__':main()
