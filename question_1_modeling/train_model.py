"""Train and validate an ABS baseline using only pre-decision features."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs' / 'xgboost-pitcher-hand'
FEATURES = ['catcher_id', 'balls', 'strikes', 'challenger_score_diff',
            'batter_hand', 'inning', 'plate_x',
            'plate_z', 'pitch_result', 'break_x', 'pitcher_hand']
CATEGORICAL = ['catcher_id', 'batter_hand', 'pitch_result', 'pitcher_hand']
SEED = 42


def challenger_score_diff(frame):
    top = frame.is_top_inning.astype(str).str.lower().eq('true')
    if not frame.pitch_result.isin(['ball', 'called_strike']).all():
        raise ValueError('Unexpected original pitch call')
    # Top: home pitches, away bats. Bottom: away pitches, home bats.
    home_challenges = (frame.pitch_result.eq('ball') & top) | (frame.pitch_result.eq('called_strike') & ~top)
    home_diff = frame.pre_score_home - frame.pre_score_away
    return pd.Series(np.where(home_challenges, home_diff, -home_diff), index=frame.index)


def features(frame, categories):
    frame = frame.assign(challenger_score_diff=challenger_score_diff(frame))
    x = frame[FEATURES].copy()
    for col in CATEGORICAL:
        x[col] = pd.Categorical(x[col].fillna('missing').astype(str),
                                categories=categories[col])
    for col in set(FEATURES) - set(CATEGORICAL):
        x[col] = pd.to_numeric(x[col], errors='coerce')
    return x


def metrics(y, p):
    return {'n': len(y), 'positive_rate': float(np.mean(y)),
            'log_loss': float(log_loss(y, p, labels=[0, 1])),
            'roc_auc': float(roc_auc_score(y, p)),
            'brier_score': float(brier_score_loss(y, p))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(ROOT / 'data-train.csv', dtype={'game_id': str})
    test = pd.read_csv(ROOT / 'data-test.csv', dtype={'game_id': str})
    categories = {col: sorted(train[col].fillna('missing').astype(str).unique())
                  for col in CATEGORICAL}
    x, xt = features(train, categories), features(test, categories)
    (OUT / 'categories.json').write_text(json.dumps(categories, indent=2) + '\n')
    fit_idx, val_idx = next(GroupShuffleSplit(n_splits=1, test_size=.2,
                          random_state=SEED).split(x, groups=train.game_id))
    assert not set(train.iloc[fit_idx].game_id) & set(train.iloc[val_idx].game_id)
    summary = {'algorithm': 'XGBoost', 'features': FEATURES, 'omitted_unavailable':
               ['batting_order_position', 'pitch_type', 'challenges_remaining'],
               'seed': SEED, 'train_games': train.iloc[fit_idx].game_id.nunique(),
               'validation_games': train.iloc[val_idx].game_id.nunique(),
               'validation_is_used_for_early_stopping': True}
    validation = train.iloc[val_idx][['play_id', 'game_id', 'is_challenge',
                                     'is_success', 'challenge_source']].copy()
    importance = []
    for target, prediction in [('is_challenge', 'p_challenge')]:
        a, b = fit_idx, val_idx
        if target != 'is_challenge':
            a = a[train.iloc[a].is_challenge.to_numpy() == 1]
            b = b[train.iloc[b].is_challenge.to_numpy() == 1]
        target_values = (train[target].eq('hitting_team').astype(int)
                         if target == 'challenge_source' else train[target])
        ya = target_values.iloc[a].astype(int)
        yb = target_values.iloc[b].astype(int)
        model = XGBClassifier(n_estimators=700, max_depth=6, learning_rate=.05,
                              objective='binary:logistic', eval_metric='logloss',
                              random_state=SEED, n_jobs=4, tree_method='hist',
                              enable_categorical=True, early_stopping_rounds=70)
        model.fit(x.iloc[a], ya, eval_set=[(x.iloc[b], yb)], verbose=100)
        p = model.predict_proba(x.iloc[b])[:, 1]
        base = np.full(len(b), ya.mean())
        summary[target] = {'model': metrics(yb, p),
                           'constant_baseline': metrics(yb, base),
                           'iterations': model.best_iteration + 1}
        validation[prediction] = model.predict_proba(x.iloc[val_idx])[:, 1]
        for feature, value in zip(FEATURES, model.feature_importances_):
            importance.append({'target': target, 'feature': feature,
                               'importance': float(value)})
        # Refit on all eligible training rows using the selected iteration count.
        eligible = train.is_challenge.eq(1) if target != 'is_challenge' else pd.Series(True, index=train.index)
        final = XGBClassifier(**{**model.get_params(),
                                 'n_estimators': model.best_iteration + 1,
                                 'early_stopping_rounds': None})
        final.fit(x.loc[eligible], target_values.loc[eligible].astype(int))
        final.save_model(str(OUT / f'{target}.ubj'))
        test[prediction] = final.predict_proba(xt)[:, 1]
        print(target, json.dumps(summary[target]), flush=True)
    test['challenge_source'] = np.where(test.pitch_result.eq('called_strike'),
                                        'hitting_team', 'pitching_team')
    # Success modeling is currently deferred at the user's request.
    for col in ['p_challenge']:
        assert test[col].between(0, 1).all() and test[col].notna().all()
    assert len(test) == 34972 and test.play_id.is_unique
    test.to_csv(OUT / 'data-test-predictions.csv', index=False)
    validation.to_csv(OUT / 'validation-predictions.csv', index=False)
    pd.DataFrame(importance).to_csv(OUT / 'feature-importance.csv', index=False)
    old_metrics = ROOT / 'outputs' / 'xgboost-break-x' / 'metrics.json'
    if old_metrics.exists():
        old_loss = json.loads(old_metrics.read_text())['is_challenge']['model']['log_loss']
        new_loss = summary['is_challenge']['model']['log_loss']
        summary['pitcher_hand_comparison'] = {'without_pitcher_hand_log_loss': old_loss,
                                      'with_pitcher_hand_log_loss': new_loss,
                                      'relative_improvement_pct': 100 * (1-new_loss/old_loss)}
    summary['challenge_source_rule'] = 'ball: pitching_team; called_strike: hitting_team'
    summary['success_model_deferred'] = True
    (OUT / 'metrics.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
