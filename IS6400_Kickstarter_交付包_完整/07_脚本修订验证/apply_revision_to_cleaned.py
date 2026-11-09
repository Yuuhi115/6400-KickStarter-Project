"""Explicitly authorized overwrite of ks_clean.csv; stage, validate, back up, replace.

This migration only updates requested features/rates and removes full-data fitted
goal caps. It retains existing rows and all unrelated fields. Missing raw history
cannot be reconstructed from the five-column dropped-row log.
"""
from pathlib import Path
import importlib.util
import hashlib
import json
import os
import shutil
import time
import numpy as np
import pandas as pd

O=Path(__file__).resolve().parent;R=O.parent
spec=importlib.util.spec_from_file_location('cleaning',R/'02_清洗规则与理由/清洗脚本.py')
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def main():
    target=R/'01_清洗数据/ks_clean.csv'
    original_hash=digest(target)
    d=pd.read_csv(target,low_memory=False)
    original_columns=d.columns.tolist()
    original_shape=d.shape
    print('Loaded',original_shape,flush=True)
    history=C.build_point_in_time_history(d)
    revised=C.fix_usd_zero_rates(d)
    for c in history:revised[c]=history[c]
    revised['goal_capped']=revised['goal']
    # A machine-readable scope prevents sample history being mistaken for lifetime history.
    revised['creator_history_scope']='retained_clean_sample'
    allowed={'creator_rank','creator_n_projects','creator_prev_success_rate','is_relaunch',
             'goal_capped','fx_rate','usd_exchange_rate','static_usd_rate'}
    pd.testing.assert_frame_equal(d[[c for c in original_columns if c not in allowed]],
                                 revised[[c for c in original_columns if c not in allowed]])
    assert revised.id.equals(d.id) and revised.shape[0]==original_shape[0]
    assert (revised.creator_n_projects==revised.creator_prior_projects+1).all()
    counts={}
    for c in ['fx_rate','usd_exchange_rate','static_usd_rate']:
        expected=d.currency.astype('string').str.strip().str.upper().eq('USD')&d[c].eq(0)
        assert (revised.loc[expected,c]==1).all()
        pd.testing.assert_series_equal(d.loc[~expected,c],revised.loc[~expected,c])
        counts[c]={'USD_zero_fixed':int(expected.sum()),
                   'non_USD_zero_retained':int((revised[c].eq(0)&revised.currency.ne('USD')).sum())}
    changes={'creator_prev_success_rate':int((~np.isclose(d.creator_prev_success_rate,revised.creator_prev_success_rate,equal_nan=True)).sum()),
             'creator_n_projects':int((d.creator_n_projects!=revised.creator_n_projects).sum()),
             'is_relaunch':int((d.is_relaunch!=revised.is_relaunch).sum()),
             'goal_capped':int((d.goal_capped!=revised.goal_capped).sum())}
    stamp=time.strftime('%Y%m%d_%H%M%S',time.gmtime())+'_UTC'
    backup_dir=O/'覆盖前备份';backup_dir.mkdir(exist_ok=True)
    backup=backup_dir/('ks_clean_before_revision_'+stamp+'.csv')
    stage=target.with_name('ks_clean.revision_staged_'+stamp+'.csv')
    revised.to_csv(stage,index=False,encoding='utf-8')
    print('Staged revised CSV',revised.shape,flush=True)
    # Read the full staged output to confirm serialization before touching original.
    loaded=pd.read_csv(stage,low_memory=False)
    assert loaded.shape==revised.shape
    assert loaded.id.equals(d.id)
    assert loaded.columns.tolist()==revised.columns.tolist()
    for c in history.columns:
        if c in ['is_relaunch','creator_history_unavailable']:
            assert loaded[c].equals(revised[c])
        else:
            assert np.allclose(loaded[c].to_numpy(dtype=float),revised[c].to_numpy(dtype=float),equal_nan=True)
    for c in ['fx_rate','usd_exchange_rate','static_usd_rate','goal_capped']:
        assert np.allclose(loaded[c],revised[c],equal_nan=True)
    assert digest(target)==original_hash,'Input changed concurrently; replacement cancelled'
    shutil.copy2(target,backup)
    assert digest(backup)==original_hash
    # All validation completed; replacement stays on the same filesystem.
    os.replace(stage,target)
    result={'status':'overwritten','dataset':str(target),'backup':str(backup),
            'shape_before':list(original_shape),'shape_after':list(revised.shape),
            'rows_added_or_removed':0,'rate_repairs':counts,'changed_rows_by_field':changes,
            'sha256_before':original_hash,'sha256_after':digest(target),
            'history_scope':'retained_clean_sample; removed campaigns absent from source',
            'unrelated_original_columns_equal':True,'staged_csv_roundtrip_validated':True}
    (O/'覆盖执行结果.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
