"""Tests only write inside a temporary directory; project CSVs are read-only."""
from pathlib import Path
import importlib.util
import json
import tempfile
import unittest
import subprocess
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location('cleaning', ROOT/'02_清洗规则与理由/清洗脚本.py')
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)

def fixture():
    t = 1700000000
    return pd.DataFrame({
        'id':list(range(10)), 'creator_id':[7]*7+[8,7,np.nan],
        'name':['same','same','other','same','late','late','exact','other','invalid','missing'],
        'state':['successful','failed','canceled','successful','failed','failed','failed','successful','failed','failed'],
        'launched_at':[t+x*3600 for x in [0,1,1,12,21,21,11,1,0,1]],
        'deadline':[t+x*3600 for x in [20,10,5,30,40,40,26,4,20,10]],
        'state_changed_at':[t+x*3600 for x in [20,11,3,31,41,41,27,4,21,11]],
    }).assign(launched_at=lambda d:d.launched_at.mask(d.id==8,0))

class RevisionTests(unittest.TestCase):
    def test_cli_and_input_overwrite_protection(self):
        validation_root=Path(__file__).resolve().parent
        assert validation_root.is_relative_to(ROOT.resolve())
        d=pd.read_csv(ROOT/'01_清洗数据/ks_clean.csv',nrows=12,low_memory=False).iloc[:,:43]
        with tempfile.TemporaryDirectory(dir=validation_root) as td:
            raw=Path(td)/'input.csv';out=Path(td)/'output.csv'
            d.to_csv(raw,index=False)
            before=raw.read_bytes()
            args=[sys.executable,str(ROOT/'02_清洗规则与理由/清洗脚本.py'),'--raw',str(raw),'--out']
            safe=subprocess.run(args+[str(out)],capture_output=True)
            self.assertEqual(safe.returncode,0,safe.stderr.decode('utf-8',errors='replace'))
            self.assertTrue(Path(td,'数据清洗文档.md').is_file())
            self.assertEqual(raw.read_bytes(),before)
            unsafe=subprocess.run(args+[str(raw)],capture_output=True)
            self.assertNotEqual(unsafe.returncode,0)
            self.assertEqual(raw.read_bytes(),before)

    def test_results_must_be_available_strictly_before_launch(self):
        d=fixture(); h=C.build_point_in_time_history(d)
        self.assertTrue(np.isnan(h.loc[1,'creator_prev_success_rate']))
        self.assertTrue(np.isnan(h.loc[6,'creator_prev_success_rate'])) # exactly at known time
        self.assertEqual(h.loc[3,'creator_prev_success_rate'],0.)
        self.assertEqual(h.loc[4,'creator_prev_success_rate'],.5)
        self.assertEqual(h.loc[4,'creator_history_known_results'],2)
        self.assertTrue(h.loc[8,'creator_history_unavailable'])
        self.assertTrue(h.loc[9,'creator_history_unavailable'])

    def test_same_time_not_history_and_first_not_relaunch(self):
        h=C.build_point_in_time_history(fixture())
        self.assertEqual(h.loc[1,'creator_prior_projects'],1)
        self.assertEqual(h.loc[2,'creator_prior_projects'],1)
        self.assertEqual(h.loc[4,'creator_prior_projects'],5)
        self.assertEqual(h.loc[5,'creator_prior_projects'],5)
        self.assertFalse(h.loc[0,'is_relaunch'])
        self.assertTrue(h.loc[1,'is_relaunch'])
        self.assertFalse(h.loc[4,'is_relaunch'])
        self.assertFalse(h.loc[5,'is_relaunch'])

    def test_future_append_and_input_order_do_not_change_past(self):
        d=fixture(); h=C.build_point_in_time_history(d)
        future=d.iloc[[0]].copy();future.index=[100];future['id']=100
        for c in ['launched_at','deadline','state_changed_at']:future[c]+=10000000
        extended=C.build_point_in_time_history(pd.concat([d,future]).sample(frac=1,random_state=42))
        pd.testing.assert_frame_equal(h,extended.loc[h.index])
        self.assertEqual(h.loc[0,'creator_n_projects'],1)

    def test_unknown_timing_results_are_not_imputed(self):
        d=fixture();d.loc[1,'state_changed_at']=np.nan
        h=C.build_point_in_time_history(d)
        self.assertTrue(np.isnan(h.loc[3,'creator_prev_success_rate']))
        d.loc[0,'state_changed_at']=d.loc[0,'launched_at']-1
        h=C.build_point_in_time_history(d)
        self.assertTrue(np.isnan(h.loc[4,'creator_prev_success_rate']))

    def test_usd_only_zero_fix_preserves_non_usd_and_raw(self):
        d=pd.DataFrame({'currency':['USD','usd','GBP','USD','USD','EUR'],
            'fx_rate':[0.,0.,0.,np.nan,-1.,1.1],
            'usd_exchange_rate':[0.,0.,0.,np.nan,-1.,1.1],
            'static_usd_rate':[1.,0.,0.,np.nan,-1.,1.1]})
        before=d.copy(deep=True); got=C.fix_usd_zero_rates(d)
        pd.testing.assert_frame_equal(d,before)
        self.assertEqual(got.fx_rate.iloc[0],1.)
        self.assertEqual(got.fx_rate.iloc[1],1.)
        self.assertEqual(got.fx_rate.iloc[2],0.)
        self.assertTrue(got.fx_rate_invalid.iloc[2])
        self.assertTrue(np.isnan(got.fx_rate.iloc[3]))
        self.assertEqual(got.fx_rate.iloc[4],-1.)
        pd.testing.assert_series_equal(got.fx_rate_raw,d.fx_rate,check_names=False)
        pd.testing.assert_frame_equal(got,C.fix_usd_zero_rates(got))

    def test_clean_pipeline_writes_only_isolated_output(self):
        d=pd.read_csv(ROOT/'01_清洗数据/ks_clean.csv',nrows=12,low_memory=False).iloc[:,:43]
        alive=pd.Series(True,index=d.index); reason=pd.Series('kept',index=d.index)
        old=(C.OUTDIR,C.DROPPED)
        try:
            validation_root=Path(__file__).resolve().parent
            assert validation_root.is_relative_to(ROOT.resolve())
            with tempfile.TemporaryDirectory(dir=validation_root) as td:
                C.OUTDIR=td;C.DROPPED=str(Path(td)/'dropped.csv.gz')
                got=C.clean_and_write(d,alive,reason,str(Path(td)/'result.csv'))
                self.assertTrue(Path(td,'result.csv').is_file())
                self.assertTrue((got.goal_capped==got.goal).all())
                self.assertTrue((got.creator_n_projects==got.creator_prior_projects+1).all())
                self.assertIn('fx_rate_raw',got.columns)
        finally:
            C.OUTDIR,C.DROPPED=old

if __name__=='__main__': unittest.main()
