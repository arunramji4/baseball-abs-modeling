"""Build and verify the Q1 test submission from the latest saved models.
Run from the workspace: .venv/bin/python question_1_modeling/build_submission.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from model_features import prepare

ROOT=Path(__file__).resolve().parent

def main(model_dir=None):
    challenge_dir=ROOT/'outputs/optuna-release'
    success_dir=ROOT/'outputs/optuna-success-hands'
    out=ROOT/'submission';out.mkdir(exist_ok=True)
    d=pd.read_csv(ROOT/'data-test.csv',dtype={'game_id':str})
    original=d.copy()
    config=json.loads((ROOT/'final_config.json').read_text())
    challenge_info=config['challenge'];success_info=config['success']
    x=prepare(d,challenge_info['features'])
    assert list(x.columns)==challenge_info['features']
    xs=prepare(d,success_info['features']);assert list(xs.columns)==success_info['features']
    challenge=XGBClassifier();challenge.load_model((model_dir or challenge_dir)/'is_challenge.ubj')
    success=XGBClassifier();success.load_model((model_dir or success_dir)/'is_success.ubj')
    d['p_challenge']=challenge.predict_proba(x)[:,1]
    d['p_success_g_challenge']=success.predict_proba(xs)[:,1]
    assert d.pitch_result.isin(['ball','called_strike']).all()
    d['challenge_source']=np.where(d.pitch_result.eq('ball'),'pitching_team','hitting_team')
    prediction_cols=['p_challenge','challenge_source','p_success_g_challenge']
    assert d.play_id.is_unique and d.play_id.equals(original.play_id)
    pd.testing.assert_frame_equal(d.drop(columns=prediction_cols),original.drop(columns=prediction_cols))
    assert d[prediction_cols].notna().all().all()
    assert np.isfinite(d[['p_challenge','p_success_g_challenge']].to_numpy()).all()
    assert d[['p_challenge','p_success_g_challenge']].ge(0).all().all()
    assert d[['p_challenge','p_success_g_challenge']].le(1).all().all()
    path=out/'data-test.csv';d.to_csv(path,index=False)
    reread=pd.read_csv(path,dtype={'game_id':str})
    assert list(reread.columns)==list(original.columns)
    assert len(reread)==len(original) and reread.play_id.equals(original.play_id)
    assert reread[prediction_cols].notna().all().all()
    report={'rows':len(d),'columns':list(d.columns),'challenge_model':str(((model_dir or challenge_dir)/'is_challenge.ubj').relative_to(ROOT)),'success_model':str(((model_dir or success_dir)/'is_success.ubj').relative_to(ROOT)),'challenge_features':challenge_info['features'],'success_features':success_info['features'],'checks':'All original rows, columns, order and non-prediction values preserved; probabilities finite and in [0,1]; no missing predictions; CSV roundtrip verified.','challenge_source':'Pitching team for balls; hitting team for called strikes.'}
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'CONTENTS.txt').write_text('data-test.csv is the completed Q1 prediction submission.\nAll three requested columns are populated. Original test rows and column order are preserved.\nRebuild with: ../.venv/bin/python build_submission.py (from question_1_modeling).\nvalidation.json records the saved models, feature lists and verification.\nThe personal process explanation must be supplied separately by the applicant.\n')
    print(path,'—',len(d),'rows verified')

if __name__=='__main__':main()
