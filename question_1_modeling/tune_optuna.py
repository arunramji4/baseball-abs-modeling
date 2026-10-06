"""Tune either or both final models, then train and build the submission CSV."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import GroupKFold,GroupShuffleSplit
from sklearn.metrics import log_loss
from xgboost import XGBClassifier
from model_features import prepare

ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['both','challenge','success'],default='both')
    parser.add_argument('--challenge-trials',type=int,default=20)
    parser.add_argument('--success-trials',type=int,default=30)
    parser.add_argument('--study-name',default='final',help='Use a new name for an independent search; existing studies resume.')
    parser.add_argument('--tune-only',action='store_true',help='Update configuration without training final models.')
    args=parser.parse_args()
    if min(args.challenge_trials,args.success_trials)<1:parser.error('Trial counts must be positive')
    config=json.loads((ROOT/'final_config.json').read_text())
    full=pd.read_csv(ROOT/'data-train.csv',dtype={'game_id':str})
    out=ROOT/'outputs/tuning';out.mkdir(exist_ok=True)
    for task in ['challenge','success']:
        if args.model not in ['both',task]:continue
        d=full if task=='challenge' else full[full.is_challenge.eq(1)].reset_index(drop=True)
        y=d['is_challenge' if task=='challenge' else 'is_success'].astype(int)
        spec=config[task];x=prepare(d,spec['features'])
        if task=='challenge':
            splits=list(GroupKFold(5,shuffle=True,random_state=config['validation']['challenge_seed']).split(d,groups=d.game_id))
        else:
            a,b=next(GroupShuffleSplit(n_splits=1,test_size=config['validation']['holdout_fraction'],random_state=config['validation']['success_holdout_seed']).split(d,groups=d.game_id))
            splits=[(a[i],a[j]) for seed in config['validation']['success_seeds'] for i,j in GroupKFold(5,shuffle=True,random_state=seed).split(d.iloc[a],groups=d.iloc[a].game_id)]
        for a,b in splits:assert not set(d.iloc[a].game_id)&set(d.iloc[b].game_id)
        study=optuna.create_study(study_name=args.study_name+'_'+task,storage='sqlite:///'+str(out/'studies.sqlite3'),load_if_exists=True,direction='minimize',sampler=optuna.samplers.TPESampler(seed=42),pruner=optuna.pruners.NopPruner())
        signature={'features':spec['features'],'validation':config['validation'],'fixed_parameters':spec['fixed_parameters']}
        previous=study.user_attrs.get('configuration')
        if previous is not None and previous!=signature:raise ValueError('Study configuration changed; choose a new --study-name')
        study.set_user_attr('configuration',signature)
        if not study.trials:study.enqueue_trial(spec['best_parameters'])
        def objective(trial):
            params={'n_estimators':trial.suggest_int('n_estimators',150,650,step=50),'max_depth':trial.suggest_int('max_depth',3 if task=='challenge' else 2,8 if task=='challenge' else 6),'learning_rate':trial.suggest_float('learning_rate',.015,.15,log=True),'min_child_weight':trial.suggest_float('min_child_weight',1,50 if task=='challenge' else 30,log=True),'subsample':trial.suggest_float('subsample',.65,1),'colsample_bytree':trial.suggest_float('colsample_bytree',.65,1),'reg_alpha':trial.suggest_float('reg_alpha',0,5 if task=='challenge' else 3),'reg_lambda':trial.suggest_float('reg_lambda',.1,30,log=True),'gamma':trial.suggest_float('gamma',0,2 if task=='challenge' else 1)}
            losses=[]
            for a,b in splits:
                m=XGBClassifier(**spec['fixed_parameters'],**params);m.fit(x.iloc[a],y.iloc[a]);losses.append(float(log_loss(y.iloc[b],m.predict_proba(x.iloc[b])[:,1])))
            trial.set_user_attr('fold_losses',losses)
            return float(np.mean(losses))
        count=args.challenge_trials if task=='challenge' else args.success_trials
        complete=sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials)
        study.optimize(objective,n_trials=max(0,count-complete))
        config[task]['best_parameters']=study.best_params
        study.trials_dataframe().to_csv(out/(task+'_trials.csv'),index=False)
        (out/(task+'_metrics.json')).write_text(json.dumps({'mean_cv_log_loss':study.best_value,'parameters':study.best_params,'features':spec['features'],'completed_trials':sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials)},indent=2)+'\n')
    (ROOT/'final_config.json').write_text(json.dumps(config,indent=2)+'\n')
    if not args.tune_only:
        subprocess.run([sys.executable,str(ROOT/'run_pipeline.py'),'--evaluate'],check=True)
if __name__=='__main__':main()
