import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
"""Generate expected remaining challenges without observed game outcomes."""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from xgboost import XGBClassifier
from train_model import ROOT, features, metrics, CATEGORICAL, SEED

OUT=ROOT/'outputs'/'estimated-remaining'


def model(n=213):
    return XGBClassifier(n_estimators=n,max_depth=6,learning_rate=.05,
        objective='binary:logistic',eval_metric='logloss',random_state=SEED,
        n_jobs=4,tree_method='hist',enable_categorical=True)


def event_predictions(d,x,reference,query):
    c=model(207);c.fit(x.iloc[reference],d.iloc[reference].is_challenge.astype(int))
    a=reference[d.iloc[reference].is_challenge.to_numpy()==1]
    s=model();s.fit(x.iloc[a],d.iloc[a].is_success.astype(int))
    return c.predict_proba(x.iloc[query])[:,1]*(1-s.predict_proba(x.iloc[query])[:,1])


def estimate(d,q):
    out=pd.Series(np.nan,index=d.index,name='estimated_challenger_challenges_remaining')
    for _,g in d.groupby('game_id',sort=False):
        states=np.array([[0.,0.,1.],[0.,0.,1.]]) # probabilities of 0,1,2 remaining
        last=1
        for r in g.sort_values(['pa_of_game','pitch_of_pa']).itertuples():
            if r.inning>last:
                for inn in range(last+1,int(r.inning)+1):
                    if inn>=10:
                        states[:,1]+=states[:,0];states[:,0]=0
                last=int(r.inning)
            top=bool(r.is_top_inning)
            home=(r.pitch_result=='ball' and top) or (r.pitch_result=='called_strike' and not top)
            team=0 if home else 1
            out.loc[r.Index]=states[team,1]+2*states[team,2]
            p=float(q.loc[r.Index]);z,o,t=states[team]
            states[team]=[z+o*p,o*(1-p)+t*p,t*(1-p)]
    return out


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(ROOT/'data-train.csv',dtype={'game_id':str})
    cats={c:sorted(d[c].fillna('missing').astype(str).unique()) for c in CATEGORICAL}
    x=features(d,cats)
    a,b=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEED).split(d,groups=d.game_id))
    q=pd.Series(np.nan,index=d.index)
    for i,j in GroupKFold(n_splits=5).split(d.iloc[a],groups=d.iloc[a].game_id):
        ref,query=a[i],a[j]
        q.iloc[query]=event_predictions(d,x,ref,query)
    q.iloc[b]=event_predictions(d,x,a,b)
    assert q.notna().all()
    remaining=estimate(d,q)
    assert remaining.between(0,2).all()
    report={'feature':'Expected remaining challenges from model-predicted unsuccessful challenge events, not exact observed balance.',
            'validation':'Game cross-fitting for training history; validation history uses models fitted on fitting games only.',
            'rules':'Initial balance two; failures consume; successes retain; zero balance replenished at extra innings.',
            'comparisons':{}}
    for label,xx in [('baseline',x),('estimated_remaining',x.join(remaining))]:
        m=model(700);m.set_params(early_stopping_rounds=70)
        m.fit(xx.iloc[a],d.iloc[a].is_challenge.astype(int),eval_set=[(xx.iloc[b],d.iloc[b].is_challenge.astype(int))],verbose=False)
        result=metrics(d.iloc[b].is_challenge.astype(int),m.predict_proba(xx.iloc[b])[:,1])
        result['iterations']=m.best_iteration+1
        result['relative_log_loss_reduction_pct']=100*(1-result['log_loss']/report['comparisons'].get('baseline',result)['log_loss'])
        report['comparisons'][label]=result
        print(label,result,flush=True)
    # Test history uses full-data helper models, never test outcomes.
    test=pd.read_csv(ROOT/'data-test.csv',dtype={'game_id':str});xt=features(test,cats)
    c=model(207);c.fit(x,d.is_challenge.astype(int))
    selected=d.is_challenge.eq(1);s=model();s.fit(x.loc[selected],d.loc[selected,'is_success'].astype(int))
    qt=pd.Series(c.predict_proba(xt)[:,1]*(1-s.predict_proba(xt)[:,1]),index=test.index)
    rt=estimate(test,qt)
    final=model(report['comparisons']['estimated_remaining']['iterations'])
    final.fit(x.join(remaining),d.is_challenge.astype(int));final.save_model(OUT/'is_challenge.ubj')
    test['p_challenge']=final.predict_proba(xt.join(rt))[:,1]
    test['challenge_source']=np.where(test.pitch_result.eq('ball'),'pitching_team','hitting_team')
    test.to_csv(OUT/'data-test-predictions.csv',index=False)
    pd.DataFrame({'play_id':test.play_id,rt.name:rt}).to_csv(OUT/'test-feature.csv',index=False)
    (OUT/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
