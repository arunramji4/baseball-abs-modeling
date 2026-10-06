"""Absolute-change sensitivity using the saved FanGraphs study panel."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / 'outputs' / 'fangraphs_discipline'


def summarize(selected, minimum):
    s = selected[selected.PA_h2 >= minimum]
    return {
        'n': len(s), 'excluded': len(selected) - len(s),
        'counts': {
            'better_than_baseline': int((s.gap_h2 < s.gap_base).sum()),
            'retained_at_least_half': int((s.retained_fraction >= .5).sum()),
            'retained_all': int((s.retained_fraction >= 1).sum()),
            'still_at_least_5pp_better': int((s.improvement_h2_pp >= 5).sum()),
            'iso_rebounded_vs_h1': int((s.ISO_h2 > s.ISO_h1).sum()),
        },
        'median_retained_fraction': float(s.retained_fraction.median()) if len(s) else None,
        'median_improvement_h2_pp': float(s.improvement_h2_pp.median()) if len(s) else None,
        'mean_gap_pp': {p: float(100*s['gap_'+p].mean()) if len(s) else None for p in ('base','h1','h2')},
    }


def main():
    df = pd.read_csv(ROOT / 'all_player_seasons.csv')
    df['improvement_h1_pp'] = 100*(df.gap_base - df.gap_h1)
    df['improvement_h2_pp'] = 100*(df.gap_base - df.gap_h2)
    # Absolute differences remain meaningful when the baseline is zero or negative.
    eligible = df[(df.PA_base >= 200) & (df.PA_h1 >= 200) & (df.ISO_h1 < df.ISO_base)]
    relative = eligible[(eligible.gap_base > 0) & (eligible.improvement_h1 > .75)]
    absolute = eligible[eligible.improvement_h1_pp >= 5]
    groups = {
        'original_relative': relative,
        'absolute_5pp': absolute,
        'absolute_5pp_positive_baseline': absolute[absolute.gap_base > 0],
        'both_thresholds': absolute[(absolute.gap_base > 0) & (absolute.improvement_h1 > .75)],
    }
    results = {}
    for name, selected in groups.items():
        selected.to_csv(ROOT / (name + '_players.csv'), index=False)
        results[name] = {
            'qualifying_player_seasons': len(selected),
            'unique_players': int(selected.playerid.nunique()),
            'outcomes': {str(m): summarize(selected,m) for m in (100,200)},
            'exclude_2021': summarize(selected[selected.year != 2021],100),
        }
    # Guard against silently changing the original cohort in this rerun.
    previous = pd.read_csv(ROOT / 'qualifying_players.csv')
    assert set(zip(relative.playerid,relative.year)) == set(zip(previous.playerid,previous.year))
    assert (absolute.improvement_h1_pp >= 5).all()
    assert (absolute.ISO_h1 < absolute.ISO_base).all()
    results['absolute_nonpositive_baseline_count'] = int((absolute.gap_base <= 0).sum())
    results['source'] = 'Cached FanGraphs panel; see study_notes.txt and raw/.'
    (ROOT / 'sensitivity_summary.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))


if __name__ == '__main__':
    main()
