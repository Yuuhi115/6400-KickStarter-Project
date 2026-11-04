from pathlib import Path
import ast,json
import numpy as np
import pandas as pd
O=Path(__file__).resolve().parent
cols=['id','state','goal','pledged','percent_funded','fx_rate','usd_exchange_rate','static_usd_rate','currency','usd_pledged','launch_dt','deadline_dt','profile_id','profile_project_id','collected_at','state_changed_at']
d=pd.read_csv(O.parent/'01_清洗数据/ks_clean.csv',usecols=cols,low_memory=False)
r={}
for c in ['fx_rate','usd_exchange_rate']:
    r[c+'_nonpositive_currency']=d.loc[d[c]<=0,'currency'].value_counts().to_dict()
    r[c+'_nonpositive_values']=d.loc[d[c]<=0,c].value_counts().to_dict()
r['profile_id_equals_project_id']=int((d.profile_id==d.profile_project_id).sum())
for key,mask in [('state_money_conflict',((d.state=='successful')&(d.pledged<d.goal))|((d.state=='failed')&(d.pledged>=d.goal))),('percent_conflict',d.percent_funded.notna()&((d.percent_funded-d.pledged/d.goal*100).abs()>.1))]:
    z=d.loc[mask,cols]
    z.to_csv(O/(key+'.csv'),index=False,encoding='utf-8-sig')
    r[key+'_states']=z.state.value_counts().to_dict()
    r[key+'_examples']=z.head(5).to_dict('records')
stats=pd.read_csv(O/'numeric_statistics.csv')
r['selected_stats']=stats.loc[stats.field.isin(['goal','pledged','usd_pledged','dur_days','percent_funded','backers_count'])].to_dict('records')
tree=ast.parse((O.parent/'02_清洗规则与理由/清洗脚本.py').read_text(encoding='utf-8'))
ns={'pd':pd,'np':np,'RULES':['R1','R2','R3','R4','R5'],'DROP_STATES':['live','submitted','started','suspended']}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['pm','funnel']],type_ignores=[]),'<audit>','exec'),ns)
toy=pd.DataFrame({'state':['failed']*5,'goal':[100,100,np.nan,np.inf,100],'pledged':[0]*5,'launched_at':[1704067200]*5,'deadline':[1704110400,1704153600,1704153600,1704153600,1704067200+365.5*86400]})
reason,alive,*_=ns['funnel'](toy)
r['synthetic_edges']=pd.DataFrame({'case':['12h_boundary','24h','missing_goal_24h','infinite_goal_24h','365.5_days'],'kept':alive,'reason':reason}).to_dict('records')
(O/'detail_summary.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(r,ensure_ascii=True,indent=2))
