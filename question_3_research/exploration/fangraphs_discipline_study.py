"""Download public FanGraphs leaderboards and measure half-season persistence."""
import json
import subprocess
from pathlib import Path
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / 'outputs' / 'fangraphs_discipline'
CACHE = ROOT / 'raw'
YEARS = [y for y in range(2016, 2026) if y != 2020]

def fetch(task):
    year, month = task
    path = CACHE / f'{year}_{month}.json'
    params = dict(age='', pos='all', stats='bat', lg='all', qual=0,
                  season=year, season1=year, startdate='', enddate='',
                  month=month, team=0, pageitems=10000, pagenum=1, ind=0, type=1)
    url = 'https://www.fangraphs.com/api/leaders/major-league/data?' + urlencode(params)
    CACHE.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        temporary=path.with_suffix('.json.part')
        try:
            subprocess.run(['curl', '-f', '-L', '--retry', '2', '--max-time', '90', '-sS', url, '-o', str(temporary)], check=True)
            downloaded=json.loads(temporary.read_text())
            if len(downloaded['data'])!=downloaded['totalCount']:
                raise ValueError(f'Incomplete leaderboard: {task}')
            if not all(int(row['Season'])==year for row in downloaded['data']):
                raise ValueError(f'Unexpected season: {task}')
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    obj = json.loads(path.read_text())
    assert len(obj['data']) == obj['totalCount'], (task, 'pagination')
    assert all(int(r['Season']) == year for r in obj['data'])
    print(year, month, obj['dateRange'], len(obj['data']), flush=True)
    return task, obj

def frame(obj):
    df = pd.DataFrame(obj['data'])
    # Traded players can have both combined and individual-team rows.
    df = df.sort_values('PA', ascending=False).drop_duplicates('playerid')
    df['gap'] = (df['SO'] - df['BB']) / df['PA'].where(df['PA']>0)
    assert np.allclose(df.loc[df.PA > 0, 'gap'],
                       (df['K%'] - df['BB%'])[df.PA > 0], atol=2e-8)
    return df[['playerid', 'PlayerName', 'PA', 'SO', 'BB', 'K%', 'BB%', 'gap', 'ISO', 'wRC+']]

def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    tasks = sorted(set([(y-1, 0) for y in YEARS] + [(y,m) for y in YEARS for m in (30,31)]))
    with ThreadPoolExecutor(max_workers=3) as pool:
        data = dict(pool.map(fetch, tasks))
    records = []
    for year in YEARS:
        b = frame(data[year-1,0]).add_suffix('_base').rename(columns={'playerid_base':'playerid'})
        h1 = frame(data[year,30]).add_suffix('_h1').rename(columns={'playerid_h1':'playerid'})
        h2 = frame(data[year,31]).add_suffix('_h2').rename(columns={'playerid_h2':'playerid'})
        joined = h1.merge(b,on='playerid',validate='one_to_one').merge(h2,on='playerid',how='left',validate='one_to_one')
        joined['year'] = year
        records.append(joined)
    allrows = pd.concat(records, ignore_index=True)
    allrows['improvement_h1'] = (allrows.gap_base-allrows.gap_h1)/allrows.gap_base.where(allrows.gap_base.ne(0))
    allrows['retained_fraction'] = (allrows.gap_base-allrows.gap_h2)/(allrows.gap_base-allrows.gap_h1).where(allrows.gap_base.ne(allrows.gap_h1))
    eligible = allrows[(allrows.PA_base>=200)&(allrows.PA_h1>=200)&(allrows.gap_base>0)]
    selected = eligible[(eligible.improvement_h1>0.75)&(eligible.ISO_h1<eligible.ISO_base)].copy()
    selected.to_csv(ROOT/'qualifying_players.csv',index=False)
    allrows.to_csv(ROOT/'all_player_seasons.csv',index=False)
    results = {'eligible_player_seasons':len(eligible),'qualifying_player_seasons':len(selected),'unique_players':int(selected.playerid.nunique()), 'outcomes':{}}
    for minimum in (1,100,200):
        s=selected[selected.PA_h2>=minimum]
        n=len(s)
        metrics={
          'better_than_baseline':s.gap_h2<s.gap_base,
          'retained_at_least_half':s.retained_fraction>=0.5,
          'retained_all':s.retained_fraction>=1,
          'still_more_than_75pct_better':s.gap_h2<0.25*s.gap_base,
          'iso_rebounded_vs_h1':s.ISO_h2>s.ISO_h1,
        }
        results['outcomes'][minimum]={'n':n,'excluded':len(selected)-n,
           'counts':{k:int(v.sum()) for k,v in metrics.items()},
           'median_retained_fraction':float(s.retained_fraction.median()) if len(s) else None,
           'mean_gap':{p:float(s['gap_'+p].mean()) if len(s) else None for p in ('base','h1','h2')}}
    # Descriptive comparison with other >=75%-improving hitters, without ISO decline.
    control=eligible[(eligible.improvement_h1>0.75)&(eligible.ISO_h1>=eligible.ISO_base)&(eligible.PA_h2>=100)]
    results['comparison_no_iso_drop']={'n':len(control),'better_than_baseline':int((control.gap_h2<control.gap_base).sum()),'retained_at_least_half':int((control.retained_fraction>=0.5).sum())}
    (ROOT/'summary.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))
    columns=['PlayerName_h1','year','PA_base','PA_h1','PA_h2','gap_base','gap_h1','gap_h2','ISO_base','ISO_h1','ISO_h2','retained_fraction']
    print(selected[columns].to_string(index=False))

if __name__ == '__main__':
    main()
