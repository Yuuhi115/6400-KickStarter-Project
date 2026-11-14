from pathlib import Path
import json
import numpy as np
import pandas as pd
O=Path(__file__).resolve().parent
cols=['id','creator_id','state','launch_dt','deadline_dt','goal','currency','static_usd_rate','usd_exchange_rate','fx_rate','usd_pledged','pledged','converted_pledged_amount','name','name_len','blurb','blurb_len','blurb_words','collected_at','creator_rank','creator_prev_success_rate','category_id','category_key','subcategory_key','is_relaunch']
d=pd.read_csv(O.parent/'01_清洗数据/ks_clean.csv',usecols=cols,low_memory=False)
d['launch']=pd.to_datetime(d.launch_dt,utc=True)
d['end']=pd.to_datetime(d.deadline_dt,utc=True)
r={}
r['years']=d['launch'].dt.year.value_counts().sort_index().to_dict()
r['name_missing_lengths']=d.loc[d.name.isna(),['id','name_len']].to_dict('records')
r['blurb_missing_lengths']=d.loc[d.blurb.isna(),['blurb_len','blurb_words']].value_counts().to_dict().__str__()
r['currency_summary']=d.groupby('currency').agg(n=('id','size'),median_goal=('goal','median'),min_rate=('static_usd_rate','min'),max_rate=('static_usd_rate','max')).reset_index().to_dict('records')
r['missing_cat_by_year']=d.assign(m=d.category_id.isna()).groupby(d['launch'].dt.year).agg(n=('id','size'),missing_n=('m','sum'),missing_pct=('m',lambda s:s.mean()*100)).reset_index().to_dict('records')
r['currency_conversion']={}
for c in ['static_usd_rate','usd_exchange_rate','fx_rate']:
    a=d.pledged*d[c];ok=d[c].notna()&(d.pledged>0)
    r['currency_conversion'][c]={'tested':int(ok.sum()),'usd_pledged_relative_error_gt_1pct':int((ok&((a-d.usd_pledged).abs()>np.maximum(1,d.usd_pledged.abs()*.01))).sum())}
for name,mask in [('odd_year',d['launch'].dt.year<2009),('duration_gt_61',(d.end-d.launch).dt.total_seconds()/86400>61),('name_missing',d.name.isna()),('largest_goal',d.goal>=d.goal.quantile(.999))]:
    d.loc[mask,cols].head(25).to_csv(O/(name+'_examples.csv'),index=False,encoding='utf-8-sig')
# Earliest later same-name campaign exposes forward-looking relaunch flags.
z=d.sort_values(['launch','id']).copy()
z['prior_same_name']=z.groupby(['creator_id','name'],dropna=False).cumcount()>0
r['relaunch_future_only_n']=int((z.is_relaunch&~z.prior_same_name).sum())
z.loc[z.is_relaunch&~z.prior_same_name,['id','creator_id','name','launch_dt']].head(20).to_csv(O/'future_relaunch_examples.csv',index=False,encoding='utf-8-sig')
(O/'extra_summary.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(r,ensure_ascii=False,indent=2))
