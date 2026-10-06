"""Location-only conditional success model, validated by entire game."""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier
from train_model import ROOT, SEED, metrics

OUT=ROOT/'outputs'/'challenge-success-location'
FEATURES=['plate_x','plate_z','sz_bot','sz_top','pitch_result']
PARAMS={'n_estimators':400,'max_depth':4,'learning_rate':.05,
        'min_child_weight':5,'reg_lambda':5,'objective':'binary:logistic',
        'eval_metric':'logloss','random_state':SEED,'n_jobs':4,
        'tree_method':'hist','enable_categorical':True}


def prepare(d):
    x=d[FEATURES].copy()
    x['pitch_result']=pd.Categorical(x.pitch_result,categories=['ball','called_strike'])
    return x


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    all_data=pd.read_csv(ROOT/'data-train.csv',dtype={'game_id':str})
    d=all_data.loc[all_data.is_challenge.eq(1)].copy().reset_index(drop=True)
    assert d.is_success.notna().all()
    x=prepare(d);y=d.is_success.astype(int)
    p=np.zeros(len(d));baseline=np.zeros(len(d));membership=np.zeros(len(d),dtype=int);scores=[]
    for fold,(a,b) in enumerate(GroupKFold(n_splits=5,shuffle=True,random_state=SEED).split(d,groups=d.game_id),1):
        assert not set(d.iloc[a].game_id)&set(d.iloc[b].game_id)
        m=XGBClassifier(**PARAMS);m.fit(x.iloc[a],y.iloc[a])
        p[b]=m.predict_proba(x.iloc[b])[:,1];baseline[b]=y.iloc[a].mean();membership[b]=fold
        scores.append({'fold':fold,**metrics(y.iloc[b],p[b])})
    final=XGBClassifier(**PARAMS);final.fit(x,y);final.save_model(OUT/'is_success.ubj')
    pd.DataFrame({'play_id':d.play_id,'game_id':d.game_id,'fold':membership,'is_success':y,'p_success_g_challenge':p}).to_csv(OUT/'out-of-fold-predictions.csv',index=False)
    t=pd.read_csv(ROOT/'data-test.csv',dtype={'game_id':str})
    prediction=pd.read_csv(ROOT/'outputs'/'optuna-strike-zone'/'data-test-predictions.csv',dtype={'game_id':str})
    assert t.play_id.equals(prediction.play_id)
    prediction['p_success_g_challenge']=final.predict_proba(prepare(t))[:,1]
    assert prediction[['p_challenge','p_success_g_challenge']].notna().all().all()
    assert prediction[['p_challenge','p_success_g_challenge']].ge(0).all().all() and prediction[['p_challenge','p_success_g_challenge']].le(1).all().all()
    prediction.to_csv(OUT/'data-test-predictions.csv',index=False)
    pd.DataFrame({'feature':FEATURES,'importance':final.feature_importances_}).to_csv(OUT/'feature-importance.csv',index=False)
    report={'features':FEATURES,'parameters':PARAMS,'challenged_training_pitches':len(d),
            'challenge_success_rate':float(y.mean()),'mean_fold_log_loss':float(np.mean([s['log_loss'] for s in scores])),
            'pooled_metrics':metrics(y,p),'constant_baseline_metrics':metrics(y,baseline),
            'fold_metrics':scores,'method':'Five game-grouped folds, fixed settings, no early stopping or tuning; trained only on challenged pitches.',
            'limitation':'Estimates success among pitches selected for challenges; not a verified ABS ruling for every unchallenged pitch.'}
    (OUT/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
