"""Validate revised helpers on all existing rows without saving revised data."""
from pathlib import Path
import importlib.util, json, hashlib
import numpy as np
import pandas as pd
O=Path(__file__).resolve().parent; R=O.parent
spec=importlib.util.spec_from_file_location('cleaning',R/'02_清洗规则与理由/清洗脚本.py')
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
cols=['id','creator_id','name','state','launched_at','deadline','state_changed_at','currency',
      'fx_rate','usd_exchange_rate','static_usd_rate','creator_rank','creator_n_projects',
      'creator_prev_success_rate','is_relaunch']
d=pd.read_csv(R/'01_清洗数据/ks_clean.csv',usecols=cols,low_memory=False)
h=C.build_point_in_time_history(d); rates=C.fix_usd_zero_rates(d)
out={'rows_validated':len(d),'checks':{},'rate_repairs':{}}
for c in ['fx_rate','usd_exchange_rate','static_usd_rate']:
    zero=d[c].eq(0);usd=d.currency.eq('USD')
    assert (rates.loc[usd&zero,c]==1).all()
    pd.testing.assert_series_equal(rates.loc[~(usd&zero),c],d.loc[~(usd&zero),c])
    out['rate_repairs'][c]={'zero_before':int(zero.sum()),'USD_zero_fixed':int((zero&usd).sum()),
        'non_USD_zero_remaining':int((zero&~usd).sum())}
assert (h.creator_n_projects==h.creator_prior_projects+1).all()
assert (h.creator_rank==h.creator_prior_projects+1).all()
# Independent brute force check: source rows with leaked history plus reproducible sample.
old=pd.read_csv(R/'06_独立清洗审核/history_availability.csv',usecols=['id'])
indices=np.union1d(d.index[d.id.isin(old.id)],np.random.default_rng(47).choice(len(d),500,replace=False))
launch=C.pm(d.launched_at).to_numpy(dtype='datetime64[ns]').view('i8')
end=C.pm(d.deadline).to_numpy(dtype='datetime64[ns]').view('i8')
changed=C.pm(d.state_changed_at).to_numpy(dtype='datetime64[ns]').view('i8')
available=np.maximum(end,changed)
y=d.state.map({'successful':1.,'failed':0.}).to_numpy()
groups=d.groupby('creator_id',sort=False).indices
for i in indices:
    j=groups[d.creator_id.iloc[i]]
    known=j[(launch[j]<launch[i]) & (available[j]<launch[i]) & np.isfinite(y[j])
            & (end[j]>launch[j]) & (changed[j]>=launch[j])]
    want=float(y[known].mean()) if len(known) else np.nan
    assert np.isclose(want,h.creator_prev_success_rate.iloc[i],equal_nan=True),(int(d.id.iloc[i]),want)
    assert len(known)==h.creator_history_known_results.iloc[i]
out['checks']['independent_history_rows']=len(indices)
out['checks']['independent_history']='passed'
out['changed_history_rate_n']=int((~np.isclose(d.creator_prev_success_rate,h.creator_prev_success_rate,equal_nan=True)).sum())
out['relaunch_changed_n']=int((d.is_relaunch!=h.is_relaunch).sum())
out['total_projects_changed_n']=int((d.creator_n_projects!=h.creator_n_projects).sum())
manifest=json.loads((R/'06_独立清洗审核/summary.json').read_text(encoding='utf-8'))
for filename,expected in manifest['hashes'].items():
    if not filename.endswith(('.csv','.gz')):continue
    digest=hashlib.sha256()
    with (R/filename).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):digest.update(block)
    assert digest.hexdigest()==expected
out['checks']['existing_data_hashes_unchanged']='passed'
out['limitation']='Original CSV absent; helper validation uses retained sample, not end-to-end raw reproduction.'
(O/'revision_validation.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=True,indent=2))
