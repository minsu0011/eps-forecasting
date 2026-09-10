"""Fail-closed static PE x forecast native TTM research scenarios."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha
from research.eps_model_lab_v1.common import atomic_csv


def gate(row):
    reasons=[]
    if not row.get('prediction_valid',True): reasons.append('EPS_PREDICTION_EXPLICITLY_INVALID')
    if row['target_key']!='ttm' or row['forecast_horizon']!=4: reasons.append('NOT_QPLUS4_NATIVE_TTM')
    if row['currency']!='USD' or row['pe_currency']!='USD': reasons.append('CURRENCY_MISMATCH')
    if row['eps_definition']!='GAAP_DILUTED' or row['pe_eps_definition']!='GAAP_DILUTED_TTM': reasons.append('EPS_DEFINITION_MISMATCH')
    if row['share_basis']!='FORECAST_ORIGIN' or row['pe_price_basis']!='contemporaneous': reasons.append('SHARE_OR_PRICE_BASIS_MISMATCH')
    if pd.isna(row['pe_date']) or pd.Timestamp(row['pe_date']).date()!=pd.Timestamp(row['asof_date']).date(): reasons.append('ASOF_DATE_MISMATCH')
    if not np.isfinite(row['predicted_eps']): reasons.append('EPS_FORECAST_UNAVAILABLE')
    elif row['predicted_eps']<=0: reasons.append('NONPOSITIVE_FORECAST_EPS_PE_INVALID_NOT_EPS_MODEL_FAILURE')
    if not np.isfinite(row['expected_pe']) or row['expected_pe']<=0: reasons.append('EXPECTED_PE_UNAVAILABLE_OR_NONPOSITIVE')
    if not np.isfinite(row['current_ttm']) or not np.isfinite(row['pe_current_ttm']): reasons.append('CURRENT_TTM_UNAVAILABLE')
    elif not np.isclose(row['current_ttm'],row['pe_current_ttm'],rtol=1e-9,atol=1e-10): reasons.append('CURRENT_TTM_ORIGIN_BASIS_MISMATCH')
    if row.get('pe_contract_failure'): reasons.append('FROZEN_PE_CONTRACT_REJECTED')
    research_valid=not reasons
    strict_reasons=reasons.copy()
    if not row.get('basis_vintage_certified',False): strict_reasons.append('SOURCE_BASIS_VINTAGE_UNCERTIFIED')
    if not row.get('eps_pretraining_pit_valid',False): strict_reasons.append('PRETRAINING_OVERLAP_UNRESOLVED')
    return {'research_scenario_valid':research_valid,'price_combo_valid':not strict_reasons,
      'research_invalid_reason':'|'.join(reasons),'strict_invalid_reason':'|'.join(strict_reasons),
      'implied_price_origin_share_basis':float(row['expected_pe']*row['predicted_eps']) if research_valid else np.nan,
      'STATIC_PE_HOLD_ASSUMPTION':True,'HORIZON_MISMATCH_LIMITATION':True,'formal_certification':False}


def run():
    lock=json.loads((RUN/'ENSEMBLE_FREEZE_V1.json').read_text(encoding='utf-8'))
    plan=next(p for p in lock['plans'] if p['track']=='local_causal' and p['target']=='ttm')
    members=plan['member_ids'];pe_root=RUN/'pe_integration';rows=[]
    for ticker in ['AAPL','MSFT','JPM','KO','CVX']:
        pe=pd.read_parquet(pe_root/'v04'/f'{ticker}.parquet').set_index('date')
        c4path=pe_root/'c4'/f'{ticker}.parquet'
        c4=pd.read_parquet(c4path).assign(date=lambda d:pd.to_datetime(d.date)).set_index('date') if c4path.exists() else None
        failure_path=pe_root/'runtime_failures'/f'{ticker}_c4_isolated.json'
        c4failure=json.loads(failure_path.read_text(encoding='utf-8'))['reason'] if failure_path.exists() else 'C4_NOT_AVAILABLE'
        for member in members:
            path=RUN/'predictions'/f'{member}.parquet'
            if sha(path)!=lock['candidate_prediction_hashes'][member]: raise RuntimeError('Frozen EPS prediction drift')
            receipt=json.loads((RUN/'model_receipts'/f'{member}.json').read_text(encoding='utf-8'))
            if receipt.get('status')!='FULL_RESEARCH_SCORED': raise RuntimeError('Frozen EPS member is no longer qualified for research combination')
            f=pd.read_parquet(path);f=f[(f.ticker==ticker)&(f.target_key=='ttm')&(f.split=='RESEARCH_TEST')]
            for r in f.to_dict('records'):
                date=pd.Timestamp(r['asof_date']).tz_localize(None).normalize();p=pe.loc[date] if date in pe.index else None
                base={k:r[k] for k in ['sample_id','ticker','asof_date','eps_model_id','predicted_eps','target_key','forecast_horizon','currency','eps_definition','share_basis','current_ttm']}
                base.update(pe_date=date if p is not None else pd.NaT,pe_currency='USD',
                  pe_eps_definition=str(p['eps_definition']) if p is not None else 'UNAVAILABLE',
                  pe_price_basis=str(p['price_basis']) if p is not None else 'UNAVAILABLE',
                  pe_current_ttm=float(p['eps_ttm']) if p is not None else np.nan,
                  origin_close=float(p['close']) if p is not None else np.nan,basis_vintage_certified=False,
                  eps_pretraining_pit_valid=bool(r['pit_valid']),ttm_target_method=r['ttm_target_method'],
                  ttm_target_approximate=bool(r['ttm_target_approximate']),
                  prediction_valid=bool(r['prediction_valid']),
                  ttm_prediction_method=r.get('ttm_prediction_method','SEE_EPS_MODEL_RECEIPT_ROW_METHOD_UNSPECIFIED'),
                  future_truth_method_fields_are_evaluation_only=True,
                  selection_surface='Five diversity-selected local causal C-lane members frozen from available-before-2022 OOF; not claimed raw-MAE top five')
                v04={**base,'pe_model_id':'v04_frozen_production_archive','expected_pe':float(p['v04_expected_pe']) if p is not None else np.nan,'pe_contract_failure':''}
                rows.append({**v04,**gate(v04)})
                c4value=float(c4.loc[date,'expected_pe']) if c4 is not None and date in c4.index else np.nan
                c4row={**base,'pe_model_id':'c4_r2_hofs_irls80_block_v04_w0500','expected_pe':c4value,
                       'pe_contract_failure':'' if np.isfinite(c4value) else c4failure}
                rows.append({**c4row,**gate(c4row)})
    output=pd.DataFrame(rows);path=RUN/'EPS_PE_STATIC_COMBINATION_V1.parquet';output.to_parquet(path,index=False)
    summary=output.groupby(['pe_model_id','eps_model_id']).agg(rows=('sample_id','size'),
       research_valid=('research_scenario_valid','sum'),strict_valid=('price_combo_valid','sum')).reset_index()
    atomic_csv(summary,RUN/'EPS_PE_COMBINATION_SMOKE.csv')
    save_json(RUN/'EPS_PE_COMBINATION_RECEIPT.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
       'members':members,'rows':len(output),'prediction_sha256':sha(path),'ensemble_freeze_sha256':sha(RUN/'ENSEMBLE_FREEZE_V1.json'),
       'research_valid':int(output.research_scenario_valid.sum()),'strict_certified_valid':int(output.price_combo_valid.sum()),
       'status':'PARTIAL_V04_REAL_SCENARIOS_C4_CONTRACT_BLOCKED','price_performance_claim':False,
       'limitations':['STATIC_PE_HOLD_ASSUMPTION','HORIZON_MISMATCH_LIMITATION','SOURCE_BASIS_VINTAGE_UNCERTIFIED'],
       'rejections':output.research_invalid_reason.value_counts().to_dict()})
    print('STATIC_PE_COMBO',len(output),'RESEARCH_VALID',int(output.research_scenario_valid.sum()),'STRICT_VALID',int(output.price_combo_valid.sum()),flush=True)


if __name__=='__main__': run()
