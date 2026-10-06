import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
"""Compare movement and velocity features for challenge occurrence only."""
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier
from train_model import features, metrics, CATEGORICAL, SEED

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs' / 'pitch-movement'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(ROOT / 'data-train.csv', dtype={'game_id': str})
    cats = {c: sorted(d[c].fillna('missing').astype(str).unique()) for c in CATEGORICAL}
    # Preserve this experiment's pre-movement baseline after break_x adoption.
    base = features(d, cats).drop(columns=['break_x'], errors='ignore')
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED)
                .split(d, groups=d.game_id))
    assert not set(d.iloc[a].game_id) & set(d.iloc[b].game_id)
    report = {'target': 'is_challenge', 'seed': SEED,
              'validation_used_for_early_stopping': True,
              'baseline_features': list(base.columns),
              'missing_values': d[['break_x', 'break_z', 'velocity']].isna().sum().to_dict(),
              'comparisons': {}}
    validation = d.iloc[b][['play_id', 'game_id', 'is_challenge']].copy()
    for label, extra in [('baseline', []), ('break_x', ['break_x']),
                         ('break_z', ['break_z']), ('velocity', ['velocity']),
                         ('all_three', ['break_x', 'break_z', 'velocity'])]:
        x = base.join(d[extra]) if extra else base
        model = XGBClassifier(n_estimators=700, max_depth=6, learning_rate=.05,
                              objective='binary:logistic', eval_metric='logloss',
                              random_state=SEED, n_jobs=4, tree_method='hist',
                              enable_categorical=True, early_stopping_rounds=70)
        model.fit(x.iloc[a], d.iloc[a].is_challenge.astype(int),
                  eval_set=[(x.iloc[b], d.iloc[b].is_challenge.astype(int))], verbose=False)
        p = model.predict_proba(x.iloc[b])[:, 1]
        result = metrics(d.iloc[b].is_challenge.astype(int), p)
        result['iterations'] = model.best_iteration + 1
        result['relative_log_loss_reduction_pct'] = 100 * (1 - result['log_loss'] / report['comparisons'].get('baseline', result)['log_loss'])
        report['comparisons'][label] = result
        validation[label] = p
        print(label, json.dumps(result), flush=True)
    (OUT / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    validation.to_csv(OUT / 'validation-predictions.csv', index=False)


if __name__ == '__main__':
    main()
