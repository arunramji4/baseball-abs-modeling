import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
"""Minimal linear logistic benchmark on the same fixed five game folds."""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
import joblib
from train_model import ROOT, SEED, metrics

OUT=ROOT/'outputs'/'logistic-core'
NUMERIC=['inning','plate_x','plate_z']
FEATURES=NUMERIC+['pitch_result']


def build():
    numeric=Pipeline([('impute',SimpleImputer(strategy='median')),
                      ('scale',StandardScaler())])
    prep=ColumnTransformer([('numeric',numeric,NUMERIC),
        ('call',OneHotEncoder(drop='if_binary',handle_unknown='ignore'),['pitch_result'])])
    return Pipeline([('features',prep),('model',LogisticRegression(C=1.,solver='lbfgs',max_iter=2000))])


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(ROOT/'data-train.csv',dtype={'game_id':str})
    t=pd.read_csv(ROOT/'data-test.csv',dtype={'game_id':str})
    x=d[FEATURES];y=d.is_challenge.astype(int)
    p=np.zeros(len(d));membership=np.zeros(len(d),dtype=int);scores=[]
    for fold,(a,b) in enumerate(GroupKFold(n_splits=5,shuffle=True,random_state=SEED).split(d,groups=d.game_id),1):
        assert not set(d.iloc[a].game_id)&set(d.iloc[b].game_id)
        m=build();m.fit(x.iloc[a],y.iloc[a]);p[b]=m.predict_proba(x.iloc[b])[:,1];membership[b]=fold
        scores.append({'fold':fold,**metrics(y.iloc[b],p[b])})
        assert m.named_steps['model'].n_iter_.max()<2000
    saved=pd.read_csv(ROOT/'outputs'/'optuna'/'folds.csv',dtype={'game_id':str})
    assert saved.play_id.equals(d.play_id) and np.array_equal(saved.fold,membership)
    final=build();final.fit(x,y)
    joblib.dump(final,OUT/'model.joblib')
    t['p_challenge']=final.predict_proba(t[FEATURES])[:,1]
    t['challenge_source']=np.where(t.pitch_result.eq('ball'),'pitching_team','hitting_team')
    t.to_csv(OUT/'data-test-predictions.csv',index=False)
    pd.DataFrame({'play_id':d.play_id,'game_id':d.game_id,'fold':membership,'is_challenge':y,'p_challenge':p}).to_csv(OUT/'out-of-fold-predictions.csv',index=False)
    names=final.named_steps['features'].get_feature_names_out()
    pd.DataFrame({'feature':names,'coefficient':final.named_steps['model'].coef_[0]}).to_csv(OUT/'coefficients.csv',index=False)
    baseline=json.loads((ROOT/'outputs'/'optuna'/'metrics.json').read_text())
    report={'features':FEATURES,'model':'LogisticRegression','C':1.,'fold_metrics':scores,
        'mean_fold_log_loss':float(np.mean([r['log_loss'] for r in scores])),
        'pooled_metrics':metrics(y,p),'tuned_xgboost_mean_fold_log_loss':baseline['best_mean_fold_log_loss'],
        'method':'Same five game-grouped folds as Optuna; imputation and scaling fitted inside each fold; no class weights.',
        'limitation':'Linear effects of raw X/Z location cannot express distance from strike-zone edges or call-specific location interactions. No nonlinear features were added.'}
    (OUT/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
