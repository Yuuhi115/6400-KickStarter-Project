from pathlib import Path
import re,json,hashlib
import numpy as np
import pandas as pd
O=Path(__file__).resolve().parent
report=(O/'数据清洗独立审核报告.md').read_text(encoding='utf-8')
assert '@@' not in report
profile=pd.read_csv(O/'column_profile.csv')
assert len(profile)==68 and profile.field.nunique()==68
ns={'pd':pd,'np':np}
section=report.split('#### F4 ')[1].split('#### F5 ')[0]
parser_code=re.findall(r'```python\n(.*?)\n```',section,re.S)[0].split('launch, deadline =')[0]
exec(parser_code,ns)
launch=pd.to_datetime(['2020-01-01','2020-01-05','2020-01-12','2020-01-12'],utc=True)
frame=pd.DataFrame({'id':[1,2,3,4],'creator_id':[7]*4,'launch_dt':launch,
    'deadline_dt':pd.to_datetime(['2020-01-10','2020-01-11','2020-02-01','2020-02-01'],utc=True),
    'state_changed_at':[1578614400,1578700800,1580515200,1580515200],
    'succ':[1.,0.,1.,0.],'name_clean':['A','A','A','A']})
ns['clean']=frame.copy()
f2=report.split('#### F2 ')[1].split('#### F3 ')[0]
exec(re.findall(r'```python\n(.*?)\n```',f2,re.S)[0],ns)
assert np.allclose(ns['clean'].creator_prev_success_rate,[np.nan,np.nan,.5,.5],equal_nan=True)
f3=report.split('#### F3 ')[1].split('#### F4 ')[0]
exec(re.findall(r'```python\n(.*?)\n```',f3,re.S)[0],ns)
assert ns['clean'].observed_prior_launches.tolist()==[0,1,2,2]
assert ns['clean'].has_prior_same_name.tolist()==[False,True,True,True]
ns['clean']=pd.DataFrame({'name_clean':[None,'Hi'],'blurb_clean':[None,'two words']})
f5=report.split('#### F5 ')[1].split('#### F6 ')[0]
exec(re.findall(r'```python\n(.*?)\n```',f5,re.S)[0],ns)
assert pd.isna(ns['clean'].blurb_len.iloc[0]) and ns['clean'].blurb_words.iloc[1]==2
summary=json.loads((O/'summary.json').read_text(encoding='utf-8'))
for name,h in summary['hashes'].items():
    hash_now=hashlib.sha256()
    with (O.parent/name).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):hash_now.update(b)
    assert h==hash_now.hexdigest(), name
result={'report_placeholders':0,'profile_columns':68,'history_snippet_test':'passed',
    'prior_launch_tie_test':'passed','text_missing_snippet_test':'passed','input_hash_recheck':'passed',
    'percentages':{'removed_pct':20223/694062*100,'missing_any_pct':629762/673839*100,
    'missing_original43_pct':582123/673839*100,'country_mismatch_pct':30882/673839*100}}
(O/'report_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=True,indent=2))
