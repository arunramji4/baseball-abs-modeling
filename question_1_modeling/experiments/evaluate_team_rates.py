import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
"""Cross-fitted offensive/defensive team challenge propensity ablations."""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from xgboost import XGBClassifier
from train_model import ROOT, features, metrics, CATEGORICAL, SEED
from experiments.evaluate_inferred_teams import learn_teams, apply_teams

OUT=ROOT/'outputs'/'team-rates'
COLS=['offence_challenge_rate','defence_challenge_rate']


def rates(reference, query):
    result=pd.DataFrame(index=query.index)
    for col,team,call,source in zip(COLS,['batting_team_id','pitching_team_id'],
                                   ['called_strike','ball'],['hitting_team','pitching_team']):
        # Denominator: pitches on which that side is eligible to challenge.
        eligible=reference.loc[reference.pitch_result.eq(call)].copy()
        eligible['event']=(eligible.is_challenge.eq(1)&eligible.challenge_source.eq(source)).astype(int)
        grouped=eligible.groupby(team,observed=True).event.agg(['sum','count'])
        prior=eligible.event.mean()
        smoothed=(grouped['sum']+200*prior)/(grouped['count']+200)
        result[col]=query[team].astype(str).map(smoothed).fillna(prior).astype(float)
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(ROOT/'data-train.csv',dtype={'game_id':str})
    a,b=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEED).split(d,groups=d.game_id))
    fit=d.iloc[a];mapping,audit=learn_teams(fit)
    tf,_,_=apply_teams(fit,fit,mapping);tv,unknown,_=apply_teams(d.iloc[b],fit,mapping)
    teams=pd.concat([tf,tv]).reindex(d.index)
    labeled=d.join(teams)
    rf=pd.DataFrame(index=d.index,columns=COLS,dtype=float)
    for i,j in GroupKFold(n_splits=5).split(fit,groups=fit.game_id):
        ref,query=labeled.iloc[a[i]],labeled.iloc[a[j]]
        assert not set(ref.game_id)&set(query.game_id)
        rf.loc[query.index]=rates(ref,query)
    rf.loc[d.iloc[b].index]=rates(labeled.iloc[a],labeled.iloc[b])
    assert rf.notna().all().all() and rf.ge(0).all().all() and rf.le(1).all().all()
    cats={c:sorted(d[c].fillna('missing').astype(str).unique()) for c in CATEGORICAL}
    base=features(d,cats)
    ids=teams.copy()
    for c in ids:
        ids[c]=pd.Categorical(ids[c],categories=sorted(set(tf[c])|{'unknown'}))
    report={'rate_definition':'Team challenges divided by eligible called pitches: offence on called strikes, defence on balls; 200-pitch prior smoothing.',
            'validation':'80/20 game split seed 42; five-fold game cross-fitting for training rates; validation rates use fitting games only.',
            'limitation':'Other-game tendencies, not chronological season-to-date; team matching uses whole-game observed rosters.',
            'comparisons':{}}
    variants=[('baseline',base),('team_ids',base.join(ids)),
              ('offence_rate',base.join(rf[[COLS[0]]])),
              ('defence_rate',base.join(rf[[COLS[1]]])),
              ('both_rates',base.join(rf)),('team_ids_and_rates',base.join(ids).join(rf))]
    for label,x in variants:
        m=XGBClassifier(n_estimators=700,max_depth=6,learning_rate=.05,
            objective='binary:logistic',eval_metric='logloss',random_state=SEED,
            n_jobs=4,tree_method='hist',enable_categorical=True,early_stopping_rounds=70)
        m.fit(x.iloc[a],d.iloc[a].is_challenge.astype(int),eval_set=[(x.iloc[b],d.iloc[b].is_challenge.astype(int))],verbose=False)
        result=metrics(d.iloc[b].is_challenge.astype(int),m.predict_proba(x.iloc[b])[:,1])
        result['iterations']=m.best_iteration+1
        result['relative_log_loss_reduction_pct']=100*(1-result['log_loss']/report['comparisons'].get('baseline',result)['log_loss'])
        report['comparisons'][label]=result
        print(label,result,flush=True)
    pd.concat([d[['play_id']],rf],axis=1).to_csv(OUT/'cross-fitted-rates.csv',index=False)
    (OUT/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
