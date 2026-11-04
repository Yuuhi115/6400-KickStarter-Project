"""Read-only audit of supplied artifacts. Writes only to this audit directory.
Run with Python 3.11+, pandas 3.0.1, numpy 2.3.5. No cleaning outputs overwritten.
"""
from pathlib import Path
import hashlib, json, platform, importlib.util
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
def save(frame, name):
    frame.to_csv(OUT / name, index=False, encoding='utf-8-sig')
def mixed(s):
    t = s.astype('string').str.replace(' UTC', '', regex=False).str.strip()
    n = pd.to_numeric(t, errors='coerce')
    v = pd.to_datetime(n, unit='s', errors='coerce', utc=True)
    m = n.isna()
    v.loc[m] = pd.to_datetime(t.loc[m], errors='coerce', format='mixed', utc=True)
    return v
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()

def main():
    p = ROOT / '01_清洗数据/ks_clean.csv'
    d = pd.read_csv(p, low_memory=False)
    removed = pd.read_csv(ROOT / '01_清洗数据/dropped_rows.csv.gz')
    N = len(d)
    print('Loaded', d.shape, flush=True)
    profile = pd.DataFrame({'field':d.columns, 'dtype':d.dtypes.astype(str).values,
        'missing_n':d.isna().sum().values, 'missing_pct':d.isna().mean().values*100,
        'unique_nonmissing':d.nunique().values})
    save(profile, 'column_profile.csv')
    nums = d.select_dtypes(include='number')
    stats = nums.describe(percentiles=[.001,.01,.05,.25,.5,.75,.95,.99,.999]).T
    stats['IQR'] = stats['75%']-stats['25%']
    stats['lower_fence'] = stats['25%']-1.5*stats['IQR']
    stats['upper_fence'] = stats['75%']+1.5*stats['IQR']
    stats['iqr_outlier_n'] = [int(((nums[c]<stats.loc[c,'lower_fence'])|(nums[c]>stats.loc[c,'upper_fence'])).sum()) for c in nums]
    save(stats.reset_index(names='field'), 'numeric_statistics.csv')
    cats = ['state','currency','country','country_final','location_country','location_type','usd_type',
            'profile_state','category_parent_name','category_name','category_key','subcategory_key',
            'spotlight','staff_pick','is_terminal_pair','is_relaunch','country_mismatch','goal_under_1']
    freq=[]
    for c in cats:
        for val,n in d[c].value_counts(dropna=False).items():
            freq.append({'field':c,'value':'<MISSING>' if pd.isna(val) else str(val),'n':int(n),'pct':n/N*100})
    save(pd.DataFrame(freq), 'category_frequencies.csv')
    checks=[]; samples=[]
    def check(key, mask, cols=None):
        mask=pd.Series(mask,index=d.index).fillna(False)
        checks.append({'check':key,'n':int(mask.sum()),'pct':float(mask.mean()*100)})
        if mask.any():
            x=d.loc[mask, cols or ['id','state','goal','currency','launch_dt','deadline_dt']].head(12).copy()
            x.insert(0,'check',key); samples.append(x)
    launch=mixed(d.launched_at); end=mixed(d.deadline); created=mixed(d.created_at)
    change=mixed(d.state_changed_at); pc=mixed(d.profile_state_changed_at)
    dur=(end-launch).dt.total_seconds()/86400
    check('full_duplicate_excess',d.duplicated())
    check('original_43_duplicate_excess',d.iloc[:,:43].duplicated())
    check('id_duplicate_excess',d.duplicated('id'))
    check('project_url_duplicate_all',d.duplicated('project_url',keep=False),['id','project_url','name','creator_id'])
    check('name_creator_duplicate_all',d.duplicated(['name','creator_id'],keep=False),['id','name','creator_id','is_relaunch','launch_dt'])
    check('name_creator_duplicate_excess',d.duplicated(['name','creator_id']))
    check('normalized_name_creator_duplicate_all',d.assign(k=d.name_clean.astype('string').str.casefold().str.replace(r'\s+',' ',regex=True)).duplicated(['k','creator_id'],keep=False))
    for c in ['goal','pledged','usd_pledged','converted_pledged_amount','backers_count','percent_funded']:
        check(c+'_negative',d[c]<0,[ 'id','state',c,'currency'])
    for c in ['fx_rate','static_usd_rate','usd_exchange_rate']:
        check(c+'_nonpositive',d[c]<=0,['id','currency',c])
    check('goal_nonpositive',d.goal<=0)
    check('goal_under_1',d.goal<1)
    check('non_USD',d.currency!='USD')
    check('duration_nonpositive',dur<=0)
    check('duration_over_365',dur>365)
    check('duration_fraction_lost',(dur-d.dur_days).abs()>1e-9,['id','dur_days','launch_dt','deadline_dt'])
    check('duration_over_60_after_policy', (dur>60+1/86400)&(launch>=pd.Timestamp('2011-06-18',tz='UTC')))
    check('duration_over_90_before_policy',(dur>90+1/86400)&(launch<pd.Timestamp('2011-06-18',tz='UTC')))
    check('duration_over_61_after_policy',(dur>61)&(launch>=pd.Timestamp('2011-06-18',tz='UTC')))
    for name,v in [('launch',launch),('deadline',end),('created',created),('state_changed',change),('profile_state_changed',pc)]:
        check(name+'_unparseable',v.isna() & d[{'launch':'launched_at','deadline':'deadline','created':'created_at','state_changed':'state_changed_at','profile_state_changed':'profile_state_changed_at'}[name]].notna())
        check(name+'_future',v>=pd.Timestamp('2026-10-04',tz='UTC'))
    check('created_after_launch',created>launch)
    check('state_change_before_launch',change<launch,['id','state','state_changed_at','launch_dt','deadline_dt'])
    check('terminal_state_change_before_deadline',d.is_terminal_pair & (change<end),['id','state','state_changed_at','launch_dt','deadline_dt'])
    batch=pd.to_datetime(d.collected_at.astype(str),format='%Y%m',errors='coerce',utc=True)
    batch_next=batch+pd.offsets.MonthBegin(1)
    check('launch_after_collection_month',launch>=batch_next)
    check('terminal_deadline_after_collection_month',d.is_terminal_pair & (end>=batch_next))
    check('state_change_after_collection_month',change>=batch_next)
    check('backers_noninteger',d.backers_count%1!=0)
    check('positive_pledged_zero_backers',(d.pledged>0)&(d.backers_count==0),['id','state','pledged','backers_count','currency'])
    check('zero_pledged_positive_backers',(d.pledged==0)&(d.backers_count>0),['id','state','pledged','backers_count','currency'])
    check('successful_below_goal',(d.state=='successful')&(d.pledged<d.goal),['id','state','goal','pledged','currency'])
    check('failed_at_or_above_goal',(d.state=='failed')&(d.pledged>=d.goal),['id','state','goal','pledged','currency'])
    calc=d.pledged/d.goal*100
    check('percent_funded_mismatch_gt_0.1',d.percent_funded.notna()&((d.percent_funded-calc).abs()>0.1),['id','goal','pledged','percent_funded','currency'])
    check('profile_project_id_diff_id',d.profile_project_id.notna()&(d.profile_project_id!=d.id),['id','profile_id','profile_project_id'])
    for c in cats:
        s=d[c].astype('string'); check(c+'_outer_whitespace',s.ne(s.str.strip()),['id',c])
        check(c+'_placeholder_token',s.str.lower().isin(['na','n/a','null','none','nan','unknown','']),['id',c])
    def ne(a,b): return a.fillna('<NA>').astype(str)!=b.fillna('<NA>').astype(str)
    check('category_key_wrong',ne(d.category_key,d.category_parent_name.astype('string').str.strip().str.lower().fillna('unknown')))
    check('subcategory_key_wrong',ne(d.subcategory_key,d.category_name.astype('string').str.strip().str.lower().fillna('unknown')))
    check('country_final_wrong',ne(d.country_final,d.country.fillna(d.location_country)))
    check('country_mismatch_wrong',d.country_mismatch!=(d.country.notna()&d.location_country.notna()&(d.country.astype(str).str[:2]!=d.location_country.astype(str).str[:2])))
    check('launch_dt_wrong',mixed(d.launch_dt)!=launch)
    check('deadline_dt_wrong',mixed(d.deadline_dt)!=end)
    check('dur_days_wrong',d.dur_days!=np.floor(dur))
    for c,v in [('yr',launch.dt.year),('mon',launch.dt.month),('dow',launch.dt.dayofweek),('hour',launch.dt.hour)]: check(c+'_wrong',d[c]!=v)
    check('log_goal_wrong',~np.isclose(d.log_goal,np.log10(d.goal)))
    cap=d.goal.quantile(.999)
    check('goal_capped_wrong',~np.isclose(d.goal_capped,d.goal.clip(upper=cap)))
    check('goal_capped_changed',d.goal!=d.goal_capped,['id','goal','goal_capped','currency'])
    check('goal_under_1_wrong',d.goal_under_1!=(d.goal<1))
    check('terminal_flag_wrong',d.is_terminal_pair!=d.state.isin(['successful','failed']))
    expected=d.state.map({'successful':1.,'failed':0.})
    check('succ_wrong',~np.isclose(d.succ,expected,equal_nan=True))
    for src,dst in [('name','name_clean'),('blurb','blurb_clean')]:
        exp=d[src].astype('string').str.replace(r'"{2,}','"',regex=True).str.strip()
        check(dst+'_wrong',ne(d[dst],exp))
        check(dst+'_remaining_quotes',d[dst].astype('string').str.contains(r'"{2,}',regex=True))
        check(dst+'_changed',ne(d[src],d[dst]),['id',src,dst])
    check('missing_blurb_wrong_lengths',d.blurb.isna()&((d.blurb_len!=0)|(d.blurb_words!=0)),['id','blurb','blurb_len','blurb_words'])
    # Artifact CSV ordering is the actual chronology used; strict availability uses max(deadline,state change).
    g=d.groupby('creator_id',sort=False)
    check('creator_rank_wrong_export_order',d.creator_rank!=g.cumcount()+1)
    check('creator_n_projects_wrong',d.creator_n_projects!=g.id.transform('size'))
    expected_prev=g.succ.transform(lambda s:s.shift().expanding().mean())
    check('prev_rate_wrong_export_order',~np.isclose(d.creator_prev_success_rate,expected_prev,equal_nan=True))
    check('relaunch_wrong',d.is_relaunch!=d.duplicated(['name','creator_id'],keep=False))
    check('future_total_projects_used',d.creator_n_projects>d.creator_rank,['id','creator_id','creator_rank','creator_n_projects','launch_dt'])
    check('first_project_marked_relaunch',d.is_relaunch&(d.creator_rank==1),['id','creator_id','creator_rank','name','is_relaunch'])
    check('same_creator_same_launch',d.duplicated(['creator_id','launch_dt'],keep=False),['id','creator_id','launch_dt','creator_rank'])
    # Event sweep per creator: a prior result is usable only after its recorded terminal state change AND deadline.
    avail=pd.concat([end,change],axis=1).max(axis=1)
    adjusted=np.full(N,np.nan); unavailable=np.zeros(N,dtype=bool); overlaps=np.zeros(N,dtype=bool)
    for _,idx in g.indices.items():
        if len(idx)<2: continue
        prior=[]
        for i in idx:
            eligible=[j for j in prior if pd.notna(d.succ.iloc[j])]
            valid=[j for j in eligible if pd.notna(avail.iloc[j]) and avail.iloc[j]<launch.iloc[i]]
            unavailable[i]=len(valid)!=len(eligible)
            overlaps[i]=any(end.iloc[j]>=launch.iloc[i] for j in eligible)
            if valid: adjusted[i]=d.succ.iloc[valid].mean()
            prior.append(i)
    check('prior_terminal_result_not_yet_available',unavailable,['id','creator_id','launch_dt','creator_prev_success_rate'])
    check('prior_terminal_campaign_not_ended',overlaps,['id','creator_id','launch_dt','creator_prev_success_rate'])
    check('prev_rate_diff_point_in_time',~np.isclose(d.creator_prev_success_rate,adjusted,equal_nan=True),['id','creator_id','creator_prev_success_rate','launch_dt'])
    save(pd.DataFrame({'id':d.id[unavailable],'creator_id':d.creator_id[unavailable],
        'launch_dt':d.launch_dt[unavailable],'current_rate':d.creator_prev_success_rate[unavailable],
        'availability_checked_rate':adjusted[unavailable]}),'history_availability.csv')
    save(pd.DataFrame(checks),'quality_checks.csv')
    if samples: save(pd.concat(samples,ignore_index=True),'suspicious_examples.csv')
    miss=[]
    for c in ['category_name','profile_blurb','blurb','usd_pledged','static_usd_rate']:
        for dim in ['yr','state','category_key','collected_at']:
            t=d.assign(m=d[c].isna()).groupby(dim).agg(n=('id','size'),missing_n=('m','sum'),missing_rate=('m','mean')).reset_index()
            t['field']=c;t['dimension']=dim;t=t.rename(columns={dim:'group'})
            miss.append(t)
    save(pd.concat(miss,ignore_index=True),'missingness_groups.csv')
    mappings=[]
    for key,label in [('category_parent_id','category_key'),('category_id','subcategory_key')]:
        t=d.groupby(key)[label].nunique(); mappings.append({'id_field':key,'multi_label_ids':int((t>1).sum())})
    # Synthetic edge tests execute only pure functions from AST; no load/write or module side effects.
    import ast
    tree=ast.parse((ROOT/'02_清洗规则与理由/清洗脚本.py').read_text(encoding='utf-8'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['pm','funnel','to_nullable_bool']]
    ns={'pd':pd,'np':np,'RULES':['R1','R2','R3','R4','R5'],'DROP_STATES':['live','submitted','started','suspended']}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<audit pure functions>','exec'),ns)
    toy=pd.DataFrame({'state':['failed']*5,'goal':[100]*5,'pledged':[0]*5,
        'launched_at':[1704067200,1704067200,1704067200,1704067200,1704067200],
        'deadline':[1704110400,1704153600,1704110400,1704110400,1704153600]})
    toy.loc[2,'goal']=np.nan;toy.loc[3,'goal']=np.inf;toy.loc[4,'goal']=np.nan
    reasons,alive,*_=ns['funnel'](toy)
    save(pd.DataFrame({'case':['12h valid campaign','24h valid campaign','missing_goal_12h','infinite_goal_12h','missing_goal_24h'],
        'kept':alive,'reason':reasons}),'synthetic_edge_tests.csv')
    summary={'shape':list(d.shape),'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__,
        'states':d.state.value_counts().to_dict(),'currency':d.currency.value_counts().to_dict(),
        'dropped_shape':list(removed.shape),'drop_reasons':removed.reason.value_counts().to_dict(),
        'dropped_states':removed.state.value_counts().to_dict(),'removed_id_unique':int(removed.id.nunique()),
        'removed_kept_id_overlap':int(removed.id.isin(d.id).sum()),
        'missing_any_n':int(d.isna().any(axis=1).sum()),'missing_any_original43_n':int(d.iloc[:,:43].isna().any(axis=1).sum()),
        'missing_blurb_n':int(d.blurb.isna().sum()),'cap_999':cap,'country_mismatch_n':int(d.country_mismatch.sum()),
        'time_range':{k:[str(v.min()),str(v.max())] for k,v in [('launch',launch),('deadline',end),('created',created),('state_changed',change)]},
        'category_mappings':mappings,'duration_fraction_summary':(dur-d.dur_days).describe().to_dict(),
        'hashes':{str(x.relative_to(ROOT)):sha(x) for x in [p,ROOT/'01_清洗数据/dropped_rows.csv.gz',ROOT/'02_清洗规则与理由/清洗脚本.py']}}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print(pd.DataFrame(checks).query('n>0').to_string(index=False),flush=True)
if __name__=='__main__': main()
