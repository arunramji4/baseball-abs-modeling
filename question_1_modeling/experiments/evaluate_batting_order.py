import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
"""Infer lineup slots using preceding plate appearances only, then ablate."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier
from train_model import features, metrics, CATEGORICAL, SEED

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs' / 'batting-order'


def infer_order(d):
    result = pd.Series(np.nan, index=d.index, name='batting_order')
    conflicts = 0
    missing_pas = 0
    for _, game in d.groupby('game_id', sort=False):
        pas = game.sort_values(['pa_of_game', 'pitch_of_pa']).drop_duplicates('pa_of_game')
        states = None
        known = {}
        previous = None
        slots = {}
        for row in pas.itertuples():
            side = 0 if row.is_top_inning else 1
            half = 2 * (row.inning - 1) + side
            pa = int(row.pa_of_game)
            if previous is None:
                missing_pas += pa - 1
                if half == 0:
                    states = {(pa % 9, 0)}
                else:
                    states = {(i % 9, (pa-i) % 9) for i in range(pa+1)}
            else:
                prev_pa, prev_half = previous
                gap = pa - prev_pa
                missing_pas += gap - 1
                if half == prev_half:
                    increments = [(gap, 0)] if side == 0 else [(0, gap)]
                else:
                    # Allocate unseen appearances across teams, including the
                    # current PA. At boundaries, retain all possible allocations.
                    increments = [(i, gap-i) for i in range(gap+1)
                                  if (i if side == 0 else gap-i) >= 1]
                states = {((a+i) % 9, (b+j) % 9)
                          for a,b in states for i,j in increments}
            key = (side, row.batter_id)
            if key in known:
                filtered = {s for s in states if (s[side] or 9) == known[key]}
                if filtered:
                    states = filtered
                else:
                    conflicts += 1
                    known.pop(key)
                    # Contradictory identity evidence invalidates the inferred
                    # counts. Re-open possibilities rather than output a slot.
                    states = {(i,j) for i in range(9) for j in range(9)}
            possibilities = {s[side] or 9 for s in states}
            if len(possibilities) == 1:
                slot = next(iter(possibilities))
                known[key] = slot
                slots[pa] = slot
            else:
                slots[pa] = np.nan
            previous = (pa, half)
        result.loc[game.index] = game.pa_of_game.map(slots)
    return result, {'known_pitch_fraction': float(result.notna().mean()),
                    'missing_plate_appearances_before_last_observed': missing_pas,
                    'identity_conflicts': conflicts}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(ROOT / 'data-train.csv', dtype={'game_id': str})
    t = pd.read_csv(ROOT / 'data-test.csv', dtype={'game_id': str})
    order, audit = infer_order(d)
    test_order, test_audit = infer_order(t)
    pd.DataFrame({'play_id':d.play_id, 'batting_order':order}).to_csv(OUT/'train-order.csv',index=False)
    pd.DataFrame({'play_id':t.play_id, 'batting_order':test_order}).to_csv(OUT/'test-order.csv',index=False)
    cats = {c: sorted(d[c].fillna('missing').astype(str).unique()) for c in CATEGORICAL}
    base = features(d, cats)
    a,b = next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEED).split(d,groups=d.game_id))
    report = {'train_reconstruction':audit,'test_reconstruction':test_audit,
              'method':'Sequential possible lineup counts, PA gaps, and previously observed batter slots. Unknown slots stay missing. No future pitches or outcome labels.',
              'comparisons':{}}
    for target in ['is_challenge','is_success']:
        ai,bi=a,b
        if target=='is_success':
            ai=ai[d.iloc[ai].is_challenge.to_numpy()==1]
            bi=bi[d.iloc[bi].is_challenge.to_numpy()==1]
        results={}
        for label,x in [('baseline',base),('batting_order',base.join(order))]:
            model=XGBClassifier(n_estimators=700,max_depth=6,learning_rate=.05,
                objective='binary:logistic',eval_metric='logloss',random_state=SEED,
                n_jobs=4,tree_method='hist',enable_categorical=True,early_stopping_rounds=70)
            model.fit(x.iloc[ai],d.iloc[ai][target].astype(int),
                eval_set=[(x.iloc[bi],d.iloc[bi][target].astype(int))],verbose=False)
            results[label]=metrics(d.iloc[bi][target].astype(int),model.predict_proba(x.iloc[bi])[:,1])
            results[label]['iterations']=model.best_iteration+1
            print(target,label,results[label],flush=True)
        results['batting_order']['relative_log_loss_reduction_pct']=100*(1-results['batting_order']['log_loss']/results['baseline']['log_loss'])
        report['comparisons'][target]=results
    (OUT/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
