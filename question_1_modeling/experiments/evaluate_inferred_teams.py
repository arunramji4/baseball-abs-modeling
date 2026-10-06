import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
"""Infer roster communities without labels; evaluate team IDs by held-out game."""
import json
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier
from train_model import ROOT, features, metrics, CATEGORICAL, SEED

OUT=ROOT/'outputs'/'inferred-teams'


def rosters(d):
    sides=defaultdict(set)
    venues={}
    for r in d.itertuples():
        top=bool(r.is_top_inning)
        defense='home' if top else 'away'
        offense='away' if top else 'home'
        game=str(r.game_id)
        # Batter/catcher IDs share a position-player namespace; pitchers separate.
        sides[(game,defense)].update(['pitcher:'+str(r.pitcher_id),'player:'+str(r.catcher_id)])
        sides[(game,offense)].add('player:'+str(r.batter_id))
        venues[(game,'home')]=str(r.venue_id)
    return sides,venues


def learn_teams(d):
    sides,venues=rosters(d)
    keys=sorted(sides)
    owners=defaultdict(list)
    for i,k in enumerate(keys):
        for player in sides[k]:owners[player].append(i)
    overlap=Counter()
    for indices in owners.values():
        for p,i in enumerate(indices):
            for j in indices[p+1:]:overlap[(min(i,j),max(i,j))]+=1
    edges=[]
    for (i,j),shared in overlap.items():
        # Require several players and a majority of the smaller observed roster.
        # One traded player cannot join two teams by themselves.
        if keys[i][0]!=keys[j][0] and shared>=4 and shared/min(len(sides[keys[i]]),len(sides[keys[j]]))>=.5:
            edges.extend([(i,j),(j,i)])
    rows=[i for i,j in edges];cols=[j for i,j in edges]
    graph=coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(keys),len(keys))).tocsr()
    n,labels=connected_components(graph,directed=False)
    mapping={k:'team_'+str(int(labels[i])) for i,k in enumerate(keys)}
    collisions=sum(mapping[(g,'home')]==mapping[(g,'away')] for g in d.game_id.unique())
    if collisions:raise ValueError(f'Roster graph merged opponents in {collisions} games')
    sizes=Counter(mapping.values())
    return mapping,{'components':n,'game_sides':len(keys),'opponent_collisions':collisions,
                    'component_game_side_sizes':sorted(sizes.values(),reverse=True)}


def apply_teams(query,reference,mapping):
    ref,_=rosters(reference);q,_=rosters(query)
    team_players=defaultdict(Counter)
    for key,players in ref.items():
        team_players[mapping[key]].update(players)
    assignments={};ambiguous=0
    for key,players in q.items():
        if key in mapping:
            assignments[key]=mapping[key];continue
        scores=sorted([(sum(p in roster for p in players),team) for team,roster in team_players.items()],reverse=True)
        if scores and scores[0][0]>=3 and (len(scores)==1 or scores[0][0]>scores[1][0]):
            assignments[key]=scores[0][1]
        else:
            assignments[key]='unknown';ambiguous+=1
    result=pd.DataFrame(index=query.index)
    home=query.game_id.map(lambda g:assignments[(str(g),'home')])
    away=query.game_id.map(lambda g:assignments[(str(g),'away')])
    result['batting_team_id']=np.where(query.is_top_inning,away,home)
    result['pitching_team_id']=np.where(query.is_top_inning,home,away)
    return result,ambiguous,assignments


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(ROOT/'data-train.csv',dtype={'game_id':str})
    t=pd.read_csv(ROOT/'data-test.csv',dtype={'game_id':str})
    a,b=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEED).split(d,groups=d.game_id))
    fit=d.iloc[a]
    mapping,audit=learn_teams(fit)
    team_fit,_,_=apply_teams(fit,fit,mapping)
    team_val,unknown,_=apply_teams(d.iloc[b],fit,mapping)
    teams=pd.concat([team_fit,team_val]).reindex(d.index)
    for c in teams:
        categories=sorted(set(team_fit[c])|{'unknown'})
        teams[c]=pd.Categorical(teams[c],categories=categories)
    cats={c:sorted(d[c].fillna('missing').astype(str).unique()) for c in CATEGORICAL}
    base=features(d,cats)
    report={'inference':audit,'unmatched_validation_sides':unknown,
            'method':'Roster overlap components learned on fitting games only; held-out sides matched by shared players. Venue recorded for audit, not used to force merges.',
            'limitation':'Uses the game roster observed across its pitches, not a strictly online pre-pitch roster. No challenge labels or dates used.',
            'comparisons':{}}
    for label,x in [('baseline',base),('inferred_teams',base.join(teams))]:
        m=XGBClassifier(n_estimators=700,max_depth=6,learning_rate=.05,
            objective='binary:logistic',eval_metric='logloss',random_state=SEED,
            n_jobs=4,tree_method='hist',enable_categorical=True,early_stopping_rounds=70)
        m.fit(x.iloc[a],d.iloc[a].is_challenge.astype(int),eval_set=[(x.iloc[b],d.iloc[b].is_challenge.astype(int))],verbose=False)
        result=metrics(d.iloc[b].is_challenge.astype(int),m.predict_proba(x.iloc[b])[:,1])
        result['iterations']=m.best_iteration+1
        result['relative_log_loss_reduction_pct']=100*(1-result['log_loss']/report['comparisons'].get('baseline',result)['log_loss'])
        report['comparisons'][label]=result
        print(label,result,flush=True)
    full_map,full_audit=learn_teams(d)
    tf,_,_=apply_teams(d,d,full_map);tt,unmatched,assignments=apply_teams(t,d,full_map)
    for c in tf:
        categories=sorted(set(tf[c])|{'unknown'})
        tf[c]=pd.Categorical(tf[c],categories=categories)
        tt[c]=pd.Categorical(tt[c],categories=categories)
    final=XGBClassifier(n_estimators=report['comparisons']['inferred_teams']['iterations'],max_depth=6,learning_rate=.05,
        objective='binary:logistic',random_state=SEED,n_jobs=4,tree_method='hist',enable_categorical=True)
    final.fit(base.join(tf),d.is_challenge.astype(int));final.save_model(OUT/'is_challenge.ubj')
    t['p_challenge']=final.predict_proba(features(t,cats).join(tt))[:,1]
    t['challenge_source']=np.where(t.pitch_result.eq('ball'),'pitching_team','hitting_team')
    t.to_csv(OUT/'data-test-predictions.csv',index=False)
    pd.concat([d[['play_id']],tf],axis=1).to_csv(OUT/'train-teams.csv',index=False)
    pd.concat([t[['play_id']],tt],axis=1).to_csv(OUT/'test-teams.csv',index=False)
    sides,venues=rosters(d)
    pd.DataFrame([{'game_id':g,'side':side,'team_id':team,'venue_id':venues.get((g,side)),
                   'observed_players':len(sides[(g,side)])} for (g,side),team in full_map.items()]).to_csv(OUT/'game-team-map.csv',index=False)
    report['full_training_inference']=full_audit;report['unmatched_test_sides']=unmatched
    (OUT/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
