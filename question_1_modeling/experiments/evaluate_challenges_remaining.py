import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
"""Observed-history experiment under the assignment-linked challenge rules."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier
from train_model import features, metrics, CATEGORICAL, SEED

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs' / 'challenges-remaining'


def remaining_challenges(d):
    values = pd.Series(np.nan, index=d.index, name='challenger_challenges_remaining')
    violations = 0
    for _, game in d.groupby('game_id', sort=False):
        balance = [2, 2]  # home, away
        last_inning = 1
        reliable = True
        for row in game.sort_values(['pa_of_game', 'pitch_of_pa']).itertuples():
            if row.inning > last_inning:
                for inning in range(last_inning+1, int(row.inning)+1):
                    if inning >= 10:
                        balance = [max(1, b) for b in balance]
                last_inning = int(row.inning)
            top = bool(row.is_top_inning)
            home_eligible = (row.pitch_result == 'ball' and top) or (row.pitch_result == 'called_strike' and not top)
            team = 0 if home_eligible else 1
            values.loc[row.Index] = balance[team] if reliable else np.nan
            if row.is_challenge == 1:
                # Attribute the observed historical event using its supplied team.
                home_actual = (row.challenge_source == 'hitting_team' and not top) or (row.challenge_source == 'pitching_team' and top)
                actual = 0 if home_actual else 1
                if balance[actual] == 0:
                    violations += 1
                    reliable = False  # future balances in this game are unknown
                if row.is_success == 0:
                    balance[actual] = max(0, balance[actual]-1)
    return values, violations


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(ROOT/'data-train.csv', dtype={'game_id':str})
    remaining, violations = remaining_challenges(d)
    cats = {c: sorted(d[c].fillna('missing').astype(str).unique()) for c in CATEGORICAL}
    base = features(d, cats)
    a,b = next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEED).split(d,groups=d.game_id))
    report = {'scope':'Requires observed prior challenges and outcomes; unavailable in supplied test games.',
              'rules_assumed':'Two initially; unsuccessful challenge consumes one; successful retains; empty balance replenished to one at each extra inning.',
              'rule_violations':violations,'known_pitch_fraction':float(remaining.notna().mean()),'comparisons':{}}
    for label,x in [('baseline',base),('challenges_remaining',base.join(remaining))]:
        model=XGBClassifier(n_estimators=700,max_depth=6,learning_rate=.05,
            objective='binary:logistic',eval_metric='logloss',random_state=SEED,
            n_jobs=4,tree_method='hist',enable_categorical=True,early_stopping_rounds=70)
        model.fit(x.iloc[a],d.iloc[a].is_challenge.astype(int),
            eval_set=[(x.iloc[b],d.iloc[b].is_challenge.astype(int))],verbose=False)
        result=metrics(d.iloc[b].is_challenge.astype(int),model.predict_proba(x.iloc[b])[:,1])
        result['relative_log_loss_reduction_pct']=100*(1-result['log_loss']/report['comparisons'].get('baseline',result)['log_loss'])
        report['comparisons'][label]=result
        print(label,result,flush=True)
    pd.DataFrame({'play_id':d.play_id,remaining.name:remaining}).to_csv(OUT/'train-feature.csv',index=False)
    (OUT/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
