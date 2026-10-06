"""Analyze configurable ISO decline and absolute K%-BB% improvement."""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / 'outputs' / 'fangraphs_discipline'
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--iso-drop', type=float, default=50)
    parser.add_argument('--gap-improvement', type=float, default=10)
    parser.add_argument('--h1-min-pa',type=int,default=200)
    args = parser.parse_args(argv)
    if not (0 < args.iso_drop <= 100 and args.gap_improvement > 0 and args.h1_min_pa > 0):
        parser.error('ISO drop must be in (0,100]; gap improvement and minimum PA must be positive')
    stem = ('iso_halved_10pp' if args.iso_drop == 50 and args.gap_improvement == 10
            else f'iso_drop_{args.iso_drop:g}pct_{args.gap_improvement:g}pp')
    if args.h1_min_pa!=200:stem+=f'_h1_{args.h1_min_pa}'
    df = pd.read_csv(ROOT / 'all_player_seasons.csv')
    df['improvement_h1_pp'] = 100 * (df.gap_base - df.gap_h1)
    df['improvement_h2_pp'] = 100 * (df.gap_base - df.gap_h2)
    selected = df[
        (df.PA_base >= 200) & (df.PA_h1 >= args.h1_min_pa) &
        (df.ISO_base > 0) & (df.ISO_h1 <= (1-args.iso_drop/100) * df.ISO_base) &
        (df.improvement_h1_pp >= args.gap_improvement)
    ].copy()
    selected.to_csv(ROOT / f'{stem}_players.csv', index=False)
    results = {'iso_drop_pct': args.iso_drop, 'gap_improvement_pp': args.gap_improvement,
               'h1_min_pa':args.h1_min_pa, 'qualifying_player_seasons': len(selected), 'outcomes': {}}
    for minimum in (100, 200):
        s = selected[selected.PA_h2 >= minimum]
        flags = {
            'better_than_baseline': s.gap_h2 < s.gap_base,
            'retained_at_least_half': s.retained_fraction >= .5,
            'retained_all': s.retained_fraction >= 1,
            'still_meets_absolute_threshold': s.improvement_h2_pp >= args.gap_improvement,
        }
        results['outcomes'][str(minimum)] = {
            'n': len(s),
            'counts': {k: int(v.sum()) for k, v in flags.items()},
            'rates': {k: float(v.mean()) if len(s) else None for k, v in flags.items()},
            'median_retained_fraction': float(s.retained_fraction.median()) if len(s) else None,
        }
    evaluated=selected[selected.PA_h2>=100].copy()
    evaluated['wrc_plus_change']=evaluated['wRC+_h2']-evaluated['wRC+_h1']
    results['groups']={}
    for name,group in [('all',evaluated),('iso_increased',evaluated[evaluated.ISO_h2>evaluated.ISO_h1]),('iso_flat_or_decreased',evaluated[evaluated.ISO_h2<=evaluated.ISO_h1])]:
        results['groups'][name]={'n':len(group),'mean_wrc_h1':float(group['wRC+_h1'].mean()) if len(group) else None,'mean_wrc_h2':float(group['wRC+_h2'].mean()) if len(group) else None,'mean_wrc_change':float(group.wrc_plus_change.mean()) if len(group) else None,'better_discipline_than_baseline':int(group.gap_h2.lt(group.gap_base).sum()),'retained_half':int(group.retained_fraction.ge(.5).sum())}
    results['qualifying']=len(selected);results['evaluated']=len(evaluated)
    evaluated.to_csv(ROOT/f'{stem}_wrc_plus_changes.csv',index=False)
    if args.iso_drop==25 and args.gap_improvement==7 and args.h1_min_pa==150:
        # Reproduce the previously exported 23-row coaching-readable cohort.
        zone_rows=[]
        for year in sorted(evaluated.year.unique()):
            for half,month in [('h1',30),('h2',31)]:
                raw=json.loads((ROOT/'raw'/f'{int(year)}_{month}.json').read_text())
                frame=pd.DataFrame(raw['data']).sort_values('PA',ascending=False).drop_duplicates('playerid')
                for row in frame[['playerid','piZone%']].itertuples(index=False,name=None):
                    zone_rows.append({'year':year,'playerid':row[0],'half':half,'zone':row[1]})
        zones=pd.DataFrame(zone_rows).pivot(index=['year','playerid'],columns='half',values='zone').rename(columns={'h1':'zone_h1','h2':'zone_h2'}).reset_index()
        evaluated=evaluated.merge(zones,on=['year','playerid'],how='left',validate='one_to_one')
        readable=pd.DataFrame({'Player':evaluated.PlayerName_h1,'Season':evaluated.year,'First_half_PA':evaluated.PA_h1,'Second_half_PA':evaluated.PA_h2,'ISO_first_half':evaluated.ISO_h1,'ISO_second_half':evaluated.ISO_h2,'ISO_improved':evaluated.ISO_h2>evaluated.ISO_h1,'K_minus_BB_first_half_pct':100*evaluated.gap_h1,'K_minus_BB_second_half_pct':100*evaluated.gap_h2,'K_minus_BB_change_pp':100*(evaluated.gap_h2-evaluated.gap_h1),'Discipline_improved_vs_first_half':evaluated.gap_h2<evaluated.gap_h1,'Discipline_better_than_previous_season':evaluated.gap_h2<evaluated.gap_base,'Zone_first_half_pct':100*evaluated.zone_h1,'Zone_second_half_pct':100*evaluated.zone_h2,'Zone_change_pp':100*(evaluated.zone_h2-evaluated.zone_h1),'Zone_increased':evaluated.zone_h2>evaluated.zone_h1,'wRC_plus_first_half':evaluated['wRC+_h1'],'wRC_plus_second_half':evaluated['wRC+_h2']})
        readable.to_csv(ROOT/'cohort_23_readable.csv',index=False)
        evaluated.to_csv(ROOT/f'{stem}_zone_and_gap.csv',index=False)
    (ROOT / f'{stem}_summary.json').write_text(json.dumps(results, indent=2))
    (ROOT / f'{stem}_notes.txt').write_text(
        'Seasons: 2016-2025, excluding 2020; previous full season is baseline.\n'
        'Uses previously downloaded FanGraphs panel; see study_notes.txt.\n'
        f'Require >=200 PA in baseline and >={args.h1_min_pa} PA in first half, positive baseline ISO,\n'
        f'first-half ISO decline >={args.iso_drop:g}%, and K%-BB% drop >={args.gap_improvement:g} percentage points.\n'
        f'Qualifying player-seasons: {len(selected)}.\n'
        'For an empty cohort, persistence rates are undefined.\n'
        'Second-half PA requirements are applied only after selecting the cohort.\n'
        '2021 uses shortened 2020 as baseline. Small samples and repeated players\n'
        'limit interpretation; these are descriptive results, not causal estimates.\n'
        + json.dumps(results, indent=2) + '\n'
    )
    print(json.dumps(results, indent=2))

if __name__=='__main__':
    main()
