"""Compare Taylor Ward's 2026 half-season Zone% change with eligible hitters."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / 'outputs' / 'zone_change_2026'
def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    frames = []
    for half in (1, 2):
        source = ROOT / f'raw_h{half}.json'
        if not source.exists():
            raise FileNotFoundError(f'Missing saved FanGraphs source: {source}')
        obj = json.loads(source.read_text())
        assert len(obj['data']) == obj['totalCount']
        if not all(int(row['Season'])==2026 for row in obj['data']):
            raise ValueError('Unexpected season in half-season source')
        frame = pd.DataFrame(obj['data'])
        assert not frame.playerid.duplicated().any()
        frames.append(frame[['playerid','PlayerName','PA','Pitches','piZone%','Zone%','pfxZone%','wRC+']])
    panel = frames[0].merge(frames[1],on='playerid',suffixes=('_h1','_h2'),validate='one_to_one')
    results = {}
    for minimum in (100, 200):
        eligible = panel[(panel.PA_h1 >= 200) & (panel.PA_h2 >= minimum)].copy()
        for metric in ('piZone%', 'Zone%', 'pfxZone%'):
            s = eligible.dropna(subset=[metric+'_h1',metric+'_h2']).copy()
            for name in [metric+'_h1',metric+'_h2']:
                if not s[name].between(0,1).all():raise ValueError(f'Invalid zone fraction: {name}')
            s['change_pp'] = 100*(s[metric+'_h2']-s[metric+'_h1'])
            ward = s[s.playerid == 17548]
            w = float(ward.change_pp.iloc[0]) if len(ward) else None
            others = s[s.playerid != 17548]
            results[f'{metric}_h2_min_{minimum}'] = {
                'n': len(s), 'increased': int((s.change_pp>0).sum()),
                'mean_change_pp': float(s.change_pp.mean()) if len(s) else None,
                'median_change_pp': float(s.change_pp.median()) if len(s) else None,
                'quantiles_pp': {str(q):float(s.change_pp.quantile(q)) if len(s) else None for q in (.1,.25,.5,.75,.9)},
                'ward_change_pp': w,
                'ward_rank_descending': int((s.change_pp>w).sum()+1) if w is not None else None,
                'other_players_with_smaller_change': int((others.change_pp<w).sum()) if w is not None else None,
                'other_players_n': len(others),
                'increased_at_least_ward': int((s.change_pp>=w).sum()) if w is not None else None,
            }
            if metric == 'piZone%':
                s.sort_values('change_pp',ascending=False).to_csv(ROOT/f'pitch_info_h2_min_{minimum}.csv',index=False)
    (ROOT/'summary.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))
    
    wrc=panel[['playerid','PlayerName_h1','PA_h1','wRC+_h1','PA_h2','wRC+_h2']].rename(columns={'PlayerName_h1':'PlayerName'})
    wrc['wrc_plus_change']=wrc['wRC+_h2']-wrc['wRC+_h1']
    wrc.to_csv(ROOT/'wrc_plus_changes.csv',index=False)

if __name__=='__main__':
    main()
