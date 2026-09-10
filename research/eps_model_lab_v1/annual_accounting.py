"""HVZ equation adaptation: annual GAAP dollar earnings to origin-share EPS.

Not an exact paper replication: SEC GAAP net income replaces Compustat IB,
cash-flow accrual proxy is NI-CFO, actual filing availability replaces fixed lag,
and pooled TRAIN-only winsorization replaces annual cross-sectional winsorization.
Only FY origins have a matching next-FY TTM target; all other origins fail closed.
"""
import json
from pathlib import Path
import sys
import time
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha
from research.eps_model_lab_v1.common import EPSAdapter,dataset,prediction_frame,save_predictions,evaluate_all,metric_row

COLS=['assets','dividends','dividend_payer','net_income','negative_earnings','accruals']


def annual_panel():
    panel=pd.read_parquet(RUN/'data/fiscal_panel.parquet')
    p=panel.loc[panel.fiscal_quarter==4].copy()
    p['dividend_payer']=np.where(p.dividends.notna(),(p.dividends>0).astype(float),np.nan)
    p['negative_earnings']=np.where(p.net_income.notna(),(p.net_income<0).astype(float),np.nan)
    p['accruals']=p.net_income-p.cfo
    future=p[['ticker','fiscal_year','net_income','asof']].copy()
    future['fiscal_year']-=1
    future=future.rename(columns={'net_income':'future_dollar_earnings','asof':'future_dollar_available'})
    return p.merge(future,on=['ticker','fiscal_year'],how='left',validate='one_to_one')


class HVZAdapter(EPSAdapter):
    def __init__(self):
        super().__init__('hvz_gaap_annual_eps_originshares',{'family':'CROSS_SECTIONAL_ANNUAL_ACCOUNTING',
         'source':'https://rodneywhitecenter.wharton.upenn.edu/wp-content/uploads/2014/03/Zhang2015Jan.pdf',
         'specification':'Appendix A equation A2, annual tau=1',
         'equation':'E_next = a + b1 Assets + b2 Dividends + b3 DividendDummy + b4 E + b5 NegativeE + b6 Accruals',
         'adaptation':'GAAP net income proxy for IB, NI-CFO accruals, actual filing timestamps, pooled training 1/99-percentile clipping',
         'conversion':'Forecast annual dollar NI / origin-known annual diluted weighted-average shares; no future shares',
         'target_scope':'Only fiscal-Q4 origins, next fiscal-year annual direct TTM EPS; no quarterly interpolation',
         'missingness':'Complete-case annual accounting; unknown dividends never replaced by zero',
         'pit_valid':True,'causal':True,'deployable':False,
         'share_assumption':'Constant origin diluted weighted-average shares; preferred/convertible numerator mismatch possible',
         'evidence_class':'RESEARCH_ONLY_ANNUAL_GAAP_ADAPTATION_NOT_PAPER_REPLICATION'})
    def prepare_data(self,frame):
        x=frame[COLS].astype(float).copy()
        for c in ['assets','dividends','net_income','accruals']: x[c]/=1e9
        return x
    def fit(self,frame,target=None):
        x=self.prepare_data(frame);y=frame.future_dollar_earnings.to_numpy()/1e9
        self.lower=x.quantile(.01);self.upper=x.quantile(.99)
        lo,hi=np.quantile(y,[.01,.99]);self.model=LinearRegression()
        self.model.fit(x.clip(self.lower,self.upper,axis=1),np.clip(y,lo,hi))
        self.fitted=True;return self
    def predict(self,frame,target=None):
        # Per original HVZ application, original un-winsorized observable predictors at inference.
        return self.model.predict(self.prepare_data(frame))*1e9/frame.diluted_shares.to_numpy()


def run():
    started=time.perf_counter();p=annual_panel();frame=dataset();outputs=[];audit=[]
    complete=p[COLS].notna().all(axis=1)&p.diluted_shares.gt(0)
    for year in range(2019,2027):
        cutoff=pd.Timestamp(f'{year}-01-01',tz='UTC')
        train=p.loc[complete&(p['asof']<cutoff)&(p['asof']>=cutoff-pd.DateOffset(years=10))&
                      (p.future_dollar_available<cutoff)&p.future_dollar_earnings.notna()]
        if len(train)<50: raise RuntimeError('Insufficient actual annual training rows')
        valid=frame[frame.asof_date.dt.year==year].copy()
        matched=valid[['sample_id','origin_accession']].merge(p.loc[complete],left_on='origin_accession',right_on='accn',how='inner',validate='one_to_one')
        a=HVZAdapter().fit(train);forecast=a.predict(matched)
        values=valid.sample_id.map(dict(zip(matched.sample_id,forecast))).to_numpy()
        path=RUN/'annual_fitted'/f'{year}.pkl';a.save(path)
        restored=HVZAdapter.load(path);np.testing.assert_allclose(forecast,restored.predict(matched),rtol=0,atol=0)
        out=prediction_frame(valid,'ttm',a.model_id,values,metadata=a.get_metadata())
        out.loc[~out.prediction_valid,'coverage_flag']='ANNUAL_SCOPE_OR_ACCOUNTING_INPUT_MISSING'
        out['ttm_prediction_method']='ANNUAL_DOLLAR_NI_DIVIDED_BY_ORIGIN_DILUTED_SHARES';outputs.append(out)
        audit.append({'year':year,'train_rows':len(train),'predictions':len(matched),
          'latest_train_origin':str(train['asof'].max()),'latest_train_label':str(train.future_dollar_available.max()),'save_load':'EXACT_PASS'})
        print(a.model_id,year,'FOLD_PASS',len(train),len(matched),flush=True)
    save_predictions(a.model_id,outputs,{'family':'CROSS_SECTIONAL_ANNUAL_ACCOUNTING','metadata':a.get_metadata(),'audit':audit,
       'runtime_seconds':time.perf_counter()-started,'smoke_status':'PASS','target_lane':['C_ANNUAL_ONLY']})
    # Report fair annual common subsets vs all already-scored models, separately from broad coverage scores.
    out=pd.concat(outputs,ignore_index=True);ids=set(out.loc[out.prediction_valid,'sample_id']);rows=[]
    for path in (RUN/'predictions').glob('*.parquet'):
        s=pd.read_parquet(path);s=s[(s.target_key=='ttm')&s.sample_id.isin(ids)]
        for split,g in s.groupby('split'):
            rows.append({'eps_model_id':path.stem,'target_key':'ttm','split':split,'mask':'HVZ_ANNUAL_ACCOUNTING_AVAILABLE_COMMON',**metric_row(g)})
    pd.DataFrame(rows).to_csv(RUN/'EPS_ANNUAL_ACCOUNTING_COMPARISON.csv',index=False)
    save_json(RUN/'smoke_receipts/hvz_gaap_annual_eps_originshares_summary.json',{'status':'PASS','save_load_all_folds':'EXACT_PASS','negative_predictions_permitted':True})
    evaluate_all()


if __name__=='__main__': run()
