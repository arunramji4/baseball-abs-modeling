import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
"""Measure observed prior-challenge history; unavailable in supplied test data."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier
from train_model import features, metrics, CATEGORICAL, SEED, FEATURES

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs' / 'challenge-history'
HISTORY = ['prior_game_challenges', 'prior_batting_team_challenges',
           'prior_pitching_team_challenges']


def prior_counts(data):
    d = data.sort_values(['game_id', 'pa_of_game', 'pitch_of_pa']).copy()
    top = d.is_top_inning.astype(str).str.lower().eq('true')
    challenged = d.is_challenge.eq(1)
    hitting = d.challenge_source.eq('hitting_team')
    pitching = d.challenge_source.eq('pitching_team')
    home = challenged & ((hitting & ~top) | (pitching & top))
    away = challenged & ((hitting & top) | (pitching & ~top))
    assert (home | away).sum() == challenged.sum()
    # Subtract the current event: every feature uses strictly earlier pitches.
    h = home.astype(int).groupby(d.game_id).cumsum() - home.astype(int)
    a = away.astype(int).groupby(d.game_id).cumsum() - away.astype(int)
    d['prior_game_challenges'] = h + a
    d['prior_batting_team_challenges'] = np.where(top, a, h)
    d['prior_pitching_team_challenges'] = np.where(top, h, a)
    assert d.groupby('game_id', sort=False).head(1)[HISTORY].eq(0).all().all()
    return d[HISTORY].reindex(data.index)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(ROOT / 'data-train.csv', dtype={'game_id': str})
    cats = {c: sorted(d[c].fillna('missing').astype(str).unique()) for c in CATEGORICAL}
    base = features(d, cats)
    counts = prior_counts(d)
    ai, bi = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED)
                  .split(d, groups=d.game_id))
    report = {'scope': 'Observed previous challenge outcomes are assumed available at prediction time; supplied test data lacks this history.',
              'features_added': HISTORY, 'comparisons': {}}
    for target in ['is_challenge', 'is_success']:
        a, b = ai, bi
        if target == 'is_success':
            a = a[d.iloc[a].is_challenge.to_numpy() == 1]
            b = b[d.iloc[b].is_challenge.to_numpy() == 1]
        results = {}
        for label, x in [('baseline', base),
                         ('game_count', base.join(counts[['prior_game_challenges']])),
                         ('game_and_team_counts', base.join(counts))]:
            model = XGBClassifier(n_estimators=700, max_depth=6, learning_rate=.05,
                                  objective='binary:logistic', eval_metric='logloss',
                                  random_state=SEED, n_jobs=4, tree_method='hist',
                                  enable_categorical=True, early_stopping_rounds=70)
            model.fit(x.iloc[a], d.iloc[a][target].astype(int),
                      eval_set=[(x.iloc[b], d.iloc[b][target].astype(int))], verbose=False)
            results[label] = metrics(d.iloc[b][target].astype(int), model.predict_proba(x.iloc[b])[:, 1])
            results[label]['iterations'] = model.best_iteration + 1
            print(target, label, results[label], flush=True)
        for label in ['game_count', 'game_and_team_counts']:
            results[label]['log_loss_change'] = results[label]['log_loss'] - results['baseline']['log_loss']
            results[label]['relative_log_loss_reduction_pct'] = 100 * (1 - results[label]['log_loss'] / results['baseline']['log_loss'])
        report['comparisons'][target] = results
    (OUT / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
