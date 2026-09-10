"""Exact C4 runtime in a clean process, without importing v04/LightGBM/SciPy.

The frozen C4 contract permits only its pinned NumPy OpenBLAS backend. Running
after v04 in the same process loads additional backends and correctly fails that
contract. Process isolation preserves, rather than relaxes, the runtime gate.
"""
from pathlib import Path
import sys
import json
import traceback
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from research.eps_model_lab_v1.pe_frozen_runtime import c4,DEST,RUN,sha,save_json
import pandas as pd
import numpy as np


if __name__=='__main__':
    binding=json.loads((DEST/'READONLY_PE_SOURCE_BINDING.json').read_text(encoding='utf-8'))
    for r in binding['live_C4_verified']:
        if sha(PROJECT/r['path'])!=r['sha256']: raise RuntimeError('Frozen C4 source drift')
    for ticker in sys.argv[1:]:
        try:
            from research.model_zoo.hierarchical_observable_fair_value_state_v7 import adapt_r4_canonical_source_v7,build_hierarchical_state_features_v7
            from research.model_zoo.hierarchical_observable_fair_value_state_v7.contracts import OFFSET_COLUMN,ALLOWED_CAUSAL_INVALID_POSITIONS_PER_ENTITY
            from research.model_zoo.hierarchical_observable_fair_value_state_v7.estimator import causal_training_valid_mask_v7
            from research.model_zoo.pe_c4_r2_numerical_robustness_v1.runner import _configure_variant
            from research.model_zoo.hierarchical_observable_fair_value_state_v7.runtime import capture_runtime_receipt_v7
            _configure_variant('c4_r2_e_irls80_block_v04')
            capture_runtime_receipt_v7(purpose='FIT')
            source=adapt_r4_canonical_source_v7(pd.read_csv(DEST/'c4_inputs'/f'{ticker}_canonical.csv'))
            state=build_hierarchical_state_features_v7(source)
            mask=causal_training_valid_mask_v7(source.observed_pe,state.features[OFFSET_COLUMN])
            bad=np.flatnonzero(~mask)
            full=pd.read_parquet(DEST/'v04'/f'{ticker}.parquet')
            fallback=full.iloc[-1800:].v04_expected_pe.to_numpy()[504:]
            save_json(DEST/'c4_domain_audit'/f'{ticker}.json',{'ticker':ticker,'runtime_gate':'PASS_EXACT_PINNED_BACKEND',
                'required_initial_invalid_positions':list(ALLOWED_CAUSAL_INVALID_POSITIONS_PER_ENTITY),
                'actual_first_fold_invalid_positions':bad[bad<504].tolist(),'all_invalid_positions':bad.tolist(),
                'v04_fallback_invalid_decision_rows':int((~np.isfinite(fallback)|(fallback<=0)).sum()),
                'policy':'No forced missingness, row deletion, positive-EPS substitution or frozen gate relaxation',
                'formal_certification':False})
            c4(ticker,full)
        except Exception as exc:
            save_json(DEST/'runtime_failures'/f'{ticker}_c4_isolated.json',{'ticker':ticker,'status':'FAILED_CLOSED',
              'reason':str(exc),'traceback':traceback.format_exc()})
            print(ticker,'ISOLATED_C4_FAILED_CLOSED',str(exc),flush=True)
            # Keep checking independent symbols; retain every fail-closed reason.
