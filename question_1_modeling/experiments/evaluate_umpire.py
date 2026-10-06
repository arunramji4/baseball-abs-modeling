import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
"""XGBoost umpire ablations with game-excluded aggregate fitting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from xgboost import XGBClassifier
from train_model import features, metrics, CATEGORICAL, SEED

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs' / 'umpire'
RATE_COLS = ['ump_challenge_rate', 'ump_overturn_given_challenge']


def rates(reference, query):
    grouped = reference.groupby('umpire_id').agg(n=('is_challenge', 'size'),
                  c=('is_challenge', 'sum'), s=('is_success', 'sum'))
    pc = reference.is_challenge.mean()
    ps = reference.loc[reference.is_challenge.eq(1), 'is_success'].mean()
    grouped[RATE_COLS[0]] = (grouped.c + 200 * pc) / (grouped.n + 200)
    grouped[RATE_COLS[1]] = (grouped.s + 20 * ps) / (grouped.c + 20)
    return pd.DataFrame({col: query.umpire_id.map(grouped[col]).fillna(prior)
                        for col, prior in zip(RATE_COLS, [pc, ps])}, index=query.index)


def game_history(d):
    s = d.sort_values(['game_id', 'pa_of_game', 'pitch_of_pa']).copy()
    keys = [s.game_id, s.umpire_id]
    c = s.is_challenge.astype(int)
    r = s.is_success.fillna(0).astype(int)
    n = s.groupby(['game_id', 'umpire_id']).cumcount()
    pc = c.groupby(keys).cumsum() - c
    pr = r.groupby(keys).cumsum() - r
    result = pd.DataFrame({'ump_game_prior_called_pitches': n,
                           'ump_game_prior_challenges': pc,
                           'ump_game_prior_overturns': pr,
                           'ump_game_prior_upheld': pc - pr}, index=s.index)
    assert result.groupby(s.game_id).head(1).eq(0).all().all()
    return result.reindex(d.index)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(ROOT / 'data-train.csv', dtype={'game_id': str})
    cats = {c: sorted(d[c].fillna('missing').astype(str).unique()) for c in CATEGORICAL}
    base = features(d, cats)
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED)
                .split(d, groups=d.game_id))
    fit = d.iloc[a]
    # Cross-fit rates by entire game. Validation outcomes never enter aggregates.
    season = pd.DataFrame(index=d.index, columns=RATE_COLS, dtype=float)
    for i, j in GroupKFold(n_splits=5).split(fit, groups=fit.game_id):
        ref, query = fit.iloc[i], fit.iloc[j]
        assert not set(ref.game_id) & set(query.game_id)
        season.loc[query.index] = rates(ref, query)
    season.loc[d.iloc[b].index] = rates(fit, d.iloc[b])
    assert season.notna().all().all()
    with_id = base.copy()
    with_id['umpire_id'] = pd.Categorical(d.umpire_id,
                            categories=sorted(fit.umpire_id.unique()))
    variants = {'baseline': base, 'umpire_id': with_id,
                'umpire_id_and_other_game_rates': with_id.join(season),
                'umpire_id_and_in_game_history': with_id.join(game_history(d)),
                'umpire_id_rates_and_in_game_history': with_id.join(season).join(game_history(d))}
    report = {'dates_available': False,
              'season_rates': 'Five-fold game-cross-fitted training rates; validation rates use fitting games only. Not chronological season-to-date.',
              'game_history': 'Strictly prior outcomes, assumed observed live; unavailable in supplied test CSV.',
              'quality_definition': 'Challenge and overturn rates; not accuracy on all calls.',
              'comparisons': {}}
    for target in ['is_challenge', 'is_success']:
        ai, bi = a, b
        if target == 'is_success':
            ai = ai[d.iloc[ai].is_challenge.to_numpy() == 1]
            bi = bi[d.iloc[bi].is_challenge.to_numpy() == 1]
        results = {}
        for label, x in variants.items():
            model = XGBClassifier(n_estimators=700, max_depth=6, learning_rate=.05,
                                  objective='binary:logistic', eval_metric='logloss',
                                  random_state=SEED, n_jobs=4, tree_method='hist',
                                  enable_categorical=True, early_stopping_rounds=70)
            model.fit(x.iloc[ai], d.iloc[ai][target].astype(int),
                      eval_set=[(x.iloc[bi], d.iloc[bi][target].astype(int))], verbose=False)
            results[label] = metrics(d.iloc[bi][target].astype(int), model.predict_proba(x.iloc[bi])[:, 1])
            results[label]['iterations'] = model.best_iteration + 1
            results[label]['relative_log_loss_reduction_pct'] = 100 * (1 - results[label]['log_loss'] / results['baseline']['log_loss'])
            print(target, label, results[label], flush=True)
        report['comparisons'][target] = results
    (OUT / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
