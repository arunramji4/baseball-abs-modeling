import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
"""Five-fold game CV for active-feature ablations and deployable candidates."""
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier
from train_model import ROOT, features, metrics, CATEGORICAL, SEED
from experiments.evaluate_batting_order import infer_order
from experiments.evaluate_inferred_teams import learn_teams, apply_teams
from experiments.evaluate_team_rates import rates as overall_rates
from experiments.evaluate_team_rates_by_inning import rates as phase_rates

OUT=ROOT/'outputs'/'five-fold'
CORE=['inning','plate_x','plate_z','pitch_result']


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(ROOT/'data-train.csv',dtype={'game_id':str})
    order,order_audit=infer_order(d)
    records=[];predictions={};started=time.time()
    for fold,(a,b) in enumerate(GroupKFold(n_splits=5,shuffle=True,random_state=SEED).split(d,groups=d.game_id),1):
        assert not set(d.iloc[a].game_id)&set(d.iloc[b].game_id)
        fit=d.iloc[a]
        cats={c:sorted(fit[c].fillna('missing').astype(str).unique()) for c in CATEGORICAL}
        base=features(d,cats)
        mapping,audit=learn_teams(fit)
        tf,_,_=apply_teams(fit,fit,mapping);tv,_,_=apply_teams(d.iloc[b],fit,mapping)
        teams=pd.concat([tf,tv]).reindex(d.index)
        labeled=d.join(teams)
        overall=pd.DataFrame(index=d.index,columns=['offence_challenge_rate','defence_challenge_rate'],dtype=float)
        phase=overall.copy()
        # Inner cross-fitting excludes the row's entire game from team rates.
        for i,j in GroupKFold(n_splits=5).split(fit,groups=fit.game_id):
            ref,query=labeled.iloc[a[i]],labeled.iloc[a[j]]
            overall.loc[query.index]=overall_rates(ref,query)
            phase.loc[query.index]=phase_rates(ref,query)
        overall.loc[d.iloc[b].index]=overall_rates(labeled.iloc[a],labeled.iloc[b])
        phase.loc[d.iloc[b].index]=phase_rates(labeled.iloc[a],labeled.iloc[b])
        assert overall.notna().all().all() and phase.notna().all().all()
        ids=teams.copy()
        for c in ids:
            ids[c]=pd.Categorical(ids[c],categories=sorted(set(tf[c])|{'unknown'}))
        ump=pd.DataFrame({'umpire_id':pd.Categorical(d.umpire_id.where(d.umpire_id.isin(fit.umpire_id.unique())),categories=sorted(fit.umpire_id.unique()))},index=d.index)
        variants={'current':base,'core_only':base[CORE]}
        for name,cols in {'catcher':['catcher_id'],'count':['balls','strikes'],
                          'score':['challenger_score_diff'],'batter_hand':['batter_hand'],
                          'pitcher_hand':['pitcher_hand'],'break_x':['break_x']}.items():
            variants['remove_'+name]=base.drop(columns=cols)
        variants.update({'add_batting_order':base.join(order),
                         'add_umpire_id':base.join(ump),
                         'add_break_z':base.join(d[['break_z']]),
                         'add_velocity':base.join(d[['velocity']]),
                         'add_break_z_velocity':base.join(d[['break_z','velocity']]),
                         'add_team_ids':base.join(ids),
                         'add_offence_rate':base.join(overall[['offence_challenge_rate']]),
                         'add_defence_rate':base.join(overall[['defence_challenge_rate']]),
                         'add_both_rates':base.join(overall),
                         'add_phase_rates':base.join(phase),
                         'add_team_ids_phase_rates':base.join(ids).join(phase)})
        for name,x in variants.items():
            # Fixed rounds: the outer evaluation fold never selects iterations.
            m=XGBClassifier(n_estimators=250,max_depth=6,learning_rate=.05,
                objective='binary:logistic',eval_metric='logloss',random_state=SEED,
                n_jobs=4,tree_method='hist',enable_categorical=True)
            m.fit(x.iloc[a],d.iloc[a].is_challenge.astype(int))
            p=m.predict_proba(x.iloc[b])[:,1]
            result=metrics(d.iloc[b].is_challenge.astype(int),p)
            records.append({'fold':fold,'variant':name,**result})
            if name not in predictions:predictions[name]=np.full(len(d),np.nan)
            predictions[name][b]=p
            print(f'fold={fold} {name} loss={result["log_loss"]:.8f} elapsed={time.time()-started:.0f}s',flush=True)
        pd.DataFrame(records).to_csv(OUT/'fold-metrics.csv',index=False)
    summary=[]
    for name,p in predictions.items():
        assert np.isfinite(p).all()
        rows=[r for r in records if r['variant']==name]
        summary.append({'variant':name,'pooled_log_loss':metrics(d.is_challenge,p)['log_loss'],
                        'mean_fold_log_loss':float(np.mean([r['log_loss'] for r in rows])),
                        'std_fold_log_loss':float(np.std([r['log_loss'] for r in rows],ddof=1))})
    baseline=next(r['pooled_log_loss'] for r in summary if r['variant']=='current')
    for row in summary:
        row['relative_improvement_pct']=100*(1-row['pooled_log_loss']/baseline)
        row['folds_better_than_current']=sum(next(r['log_loss'] for r in records if r['fold']==f and r['variant']==row['variant']) < next(r['log_loss'] for r in records if r['fold']==f and r['variant']=='current') for f in range(1,6))
    pd.DataFrame(summary).sort_values('pooled_log_loss').to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame({'play_id':d.play_id,**predictions}).to_csv(OUT/'out-of-fold-predictions.csv',index=False)
    (OUT/'method.json').write_text(json.dumps({'folds':5,'group':'game_id','seed':SEED,
        'fixed_iterations':250,'early_stopping':False,'target':'is_challenge',
        'core_features':CORE,'batting_order_audit':order_audit,
        'team_rates':'Inner game cross-fitting; held-out outcomes excluded.',
        'limitations':['Same dataset previously used for model development; not an untouched final evaluation.',
                       'Team inference uses whole-game observed rosters; no chronological dates.',
                       'Observed in-game challenge counts and umpire outcome histories excluded because unavailable in test games.']},indent=2)+'\n')
    print(pd.DataFrame(summary).sort_values('pooled_log_loss').to_string(index=False),flush=True)


if __name__=='__main__':main()
