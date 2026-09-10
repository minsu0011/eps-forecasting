"""Reconcile actual audit completion and known research limitations for handoff."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from research.eps_model_lab_v1.bootstrap import RUN, sha, save_json


def run():
    checks = []; replay_models = set()
    def check(name, passed, evidence, detail=None):
        checks.append({'check': name, 'pass': bool(passed), 'evidence': evidence, 'detail': detail})
    def read(path): return json.loads((RUN/path).read_text(encoding='utf-8'))
    integrity = read('EPS_RESULT_INTEGRITY_AUDIT.json')
    check('All saved prediction identities/truth/hash contracts', integrity['all_pass'], 'EPS_RESULT_INTEGRITY_AUDIT.json', integrity['model_count'])
    prefix = read('SEC_FUTURE_SNAPSHOT_TRUNCATION_AUDIT.json')
    check('Complete 69-company four-cutoff source-prefix audit', prefix['all_pass'] and prefix['completed_tickers'] == prefix['expected_tickers'] == 69
          and len([t for r in prefix['records'] for t in r['tests']]) == 276, 'SEC_FUTURE_SNAPSHOT_TRUNCATION_AUDIT.json')
    check('Frozen data still bound to prefix audit', prefix['samples_sha256_after'] == sha(RUN/'data/samples.parquet'), 'SEC_FUTURE_SNAPSHOT_TRUNCATION_AUDIT.json')
    groups = [*sorted((RUN/'reload_audits').glob('*.json')), *sorted((RUN/'classical_replay_audits').glob('*.json')),
              *sorted((RUN/'foundation_replay_audits').glob('*.json')), RUN/'TABULAR_FRESH_PROCESS_RELOAD_AUDIT.json',
              RUN/'ANNUAL_ACCOUNTING_FRESH_PROCESS_REPLAY.json', RUN/'ENSEMBLE_FRESH_PROCESS_REPLAY.json']
    for path in groups:
        data = json.loads(path.read_text(encoding='utf-8'))
        expected = data.get('expected', data.get('expected_artifacts'))
        passed = data.get('all_pass', data.get('all_reload_pass', False))
        check('Actual saved-model replay '+path.name, passed and data.get('completed') == expected,
              str(path.relative_to(RUN)), {'completed': data.get('completed'), 'expected': expected})
        for r in data.get('records', []):
            if r.get('model_id'): replay_models.add(r['model_id'])
        if data.get('model_id'): replay_models.add(data['model_id'])
        if path.stem == 'moment_heads': replay_models.add('moment_embedding_ridge')
    for directory, expected in [('probabilistic_replay', 40), ('additional_replay', None), ('origin_isolation_replay', 5)]:
        for path in sorted((RUN/directory).glob('*.json')):
            data = json.loads(path.read_text(encoding='utf-8'))
            count = len(data.get('records', [])); intended = expected or (40 if path.stem.startswith('tabpfn') else 5)
            check('Actual fresh point/quantile replay '+path.stem, data.get('all_pass') and count == intended,
                  str(path.relative_to(RUN)), {'completed': count, 'expected': intended})
            replay_models.add(path.stem)
    receipts = [json.loads(p.read_text(encoding='utf-8')) for p in (RUN/'model_receipts').glob('*.json')]
    qualified = {r['model_id'] for r in receipts if r.get('status') == 'FULL_RESEARCH_SCORED'}
    missing = sorted(qualified-replay_models)
    check('Every qualified scored identity has an actual replay record', not missing, 'All replay manifests', missing)
    for path, key in [('FINAL_PE_READONLY_AUDIT.json', 'all_unchanged'), ('FINAL_PRETRAINED_WEIGHT_HASH_AUDIT.json', 'all_unchanged'),
                      ('ALL_ANNUAL_AUDIT_COMPLETION_REVIEW.json', 'artifact_completion_pass'),
                      ('JOINT_ADAPTER_API_COMPLETION_AUDIT.json', 'all_pass')]:
        check(path, read(path).get(key), path)
    annual = read('ALL_ANNUAL_ORIGIN_INDEPENDENCE_AUDIT.json')
    check('All annual neural strict-FP32 probes', annual['completed_annual_artifacts'] == annual['expected_annual_artifacts'] == 280
          and annual['probe_records'] == 840 and annual['all_input_value_mutations_pass'] and annual['all_singleton_numerics_pass'],
          'ALL_ANNUAL_ORIGIN_INDEPENDENCE_AUDIT.json')
    tab = read('TABPFN_NUMERICAL_LIMITATION_REVIEW.json')
    check('TabPFN singleton failures explicitly reviewed, not silently passed', tab['completed_heads'] == 40 and len(tab['failed_singleton_heads']) == 2
          and tab['all_future_label_other_origin_and_permutation_tests_pass'] and tab['no_more_v1_retries'] and not tab['tolerance_relaxed'],
          'TABPFN_NUMERICAL_LIMITATION_REVIEW.json')
    throughput = read('CPU_SAMPLING_PARALLEL_REPLAY.json')
    check('Exact-output CPU throughput optimization', throughput['status'] == 'PASS_EXACT_ALL_ORIGINS'
          and throughput['record']['origins'] == 2101 and throughput['record']['exact']
          and not throughput['original_predictions_overwritten'], 'CPU_SAMPLING_PARALLEL_REPLAY.json')
    for base in ['lag_llama_zero_shot', 'moirai1p1_small', 'moirai_moe_small']:
        path = f'sampling_stability/{base}_origin_isolated/COMPLETION.json'
        data = read(path)
        check('Sampling sensitivity diagnostic '+base, data['status'] == 'PASS_DIAGNOSTIC_COMPLETE' and data['metric_rows'] == 8, path)
    tree = ET.parse(RUN/'FINAL_CONTRACT_TESTS.xml'); suites = list(tree.getroot().iter('testsuite'))
    tests = sum(int(x.attrib.get('tests', 0)) for x in suites)
    failures = sum(int(x.attrib.get('failures', 0))+int(x.attrib.get('errors', 0)) for x in suites)
    check('Final actual contract tests', tests > 0 and failures == 0, 'FINAL_CONTRACT_TESTS.xml', {'tests': tests, 'failures_or_errors': failures})
    lock = read('ENSEMBLE_FREEZE_V1.json')
    check('OOF-only immutable ensemble membership/weights', len(lock['plans']) == 10 and not lock['test_scores_consumed'], 'ENSEMBLE_FREEZE_V1.json')
    shortlist = read('EPS_RESEARCH_SHORTLIST_V1.json')
    check('Supported OOF shortlist', 5 <= len(shortlist['models']) <= 10 and not shortlist['roles_use_test'], 'EPS_RESEARCH_SHORTLIST_V1.json')
    portable = read('PORTABLE_EVALUATOR_PREFLIGHT.json')
    check('Actual portable evaluator preflight', portable['all_pass'], 'PORTABLE_EVALUATOR_PREFLIGHT.json')
    clean = read('CLEAN_ENV_REPLAY_QUEUE.json')
    check('Fresh pinned evaluator environment actual replay', clean['status'] == 'PASS_FRESH_PINNED_ENVIRONMENT'
          and clean['portable_evaluator_pass'] and clean['original_lock_unchanged'], 'CLEAN_ENV_REPLAY_QUEUE.json')
    independent = read('INDEPENDENT_NUMPY_METRIC_AUDIT.json')
    check('Independent NumPy score recomputation', independent['all_pass'] and independent['prediction_files'] == 117
          and independent['metric_comparisons'] == 18104, 'INDEPENDENT_NUMPY_METRIC_AUDIT.json')
    for base in ['moirai1p1_small', 'moirai_moe_small']:
        name = f'cpu_moirai_throughput/{base}_origin_isolated/COMPLETION.json'
        profile = read(name)
        check('Moirai exact CPU scheduling '+base, profile['status'] == 'PASS_EXACT_ALL_ORIGINS'
              and profile['record']['origins'] == 2101 and profile['original_predictions_unchanged'], name)
    interpretation = read('OUTCOME_INTERPRETATION_RECEIPT.json')
    check('Outcome review preserves scientific freezes', interpretation['all_frozen_artifacts_unchanged'], 'OUTCOME_INTERPRETATION_RECEIPT.json')
    recovery = read('PORTFOLIO_FINISH_RECOVERY.json')
    check('Initial portable failure explicitly recovered', recovery['status'] == 'COMPLETE_PREPACKAGE'
          and recovery['all_scientific_outputs_unchanged'], 'PORTFOLIO_FINISH_RECOVERY.json')
    quality = read('SOURCE_SHARE_UNIT_QUALITY_AUDIT.json')
    check('Confirmed share-unit defect disclosed, not silently passed', quality['status'] == 'CONFIRMED_SOURCE_UNIT_DEFECT_UNRESOLVED_IN_FROZEN_V1_3'
          and quality['all_frozen_artifacts_unchanged'] and len(quality['directly_exposed_scored_base_ids']) == 17,
          'SOURCE_SHARE_UNIT_QUALITY_AUDIT.json')
    ablation = read('share_unit_diagnostics/COMPLETION.json')
    check('Separate source-quality diagnostic actual full fit and replay', ablation['trained_heads'] == ablation['fresh_replay_heads'] == 120
          and ablation['all_original_artifacts_unchanged'] and ablation['new_portfolio_members'] == 0,
          'share_unit_diagnostics/COMPLETION.json')
    duration = read('SOURCE_ACCOUNTING_DURATION_QUALITY_AUDIT.json')
    check('Mixed-duration accounting semantics explicitly disclosed',
          duration['status'] == 'CONFIRMED_MIXED_DURATION_FEATURE_CONTRACT_DEFECT_UNRESOLVED'
          and duration['duration_mismatch_cells'] == 147 and len(duration['direct_base_ids']) == 26
          and not duration['frozen_dataset_or_predictions_changed'], 'SOURCE_ACCOUNTING_DURATION_QUALITY_AUDIT.json')
    context = read('NEXT_WAVE_CONTEXT_SELECTOR_REVIEW.json')
    check('Exact-context selector shadow-tested without replacing frozen data',
          context['status'] == 'PASS_SHADOW_ONLY_NOT_INTEGRATED' and context['compared_cells'] == 53520
          and context['source_unchanged'] and not context['new_dataset_created'] and not context['source_share_units_repaired'],
          'NEXT_WAVE_CONTEXT_SELECTOR_REVIEW.json')
    late = read('EPS_LATE_FRONTIER_REVIEW.json')
    check('Additional official architectures separately scored and replayed',
          late['status'] == 'PASS_SEPARATE_POST_FREEZE_DIAGNOSTICS' and late['additional_prediction_files'] >= 3
          and late['fresh_saved_replay_checks'] == 40*late['additional_prediction_files']
          and late['main_portfolio_unchanged'] and late['new_portfolio_members'] == 0, 'EPS_LATE_FRONTIER_REVIEW.json')
    training_seeds = read('CHRONOS_TRAINING_SEED_REVIEW.json')
    check('Five fixed OOF training seeds actually replayed, no best-seed swap',
          training_seeds['status'] == 'PASS_FIVE_SEED_OOF_TRAINING_SENSITIVITY'
          and training_seeds['actual_saved_replay_checks'] == 75
          and training_seeds['original_seed_reproduced_bit_exact'] and not training_seeds['best_seed_selected'],
          'CHRONOS_TRAINING_SEED_REVIEW.json')
    frontier_env = read('late_frontier/environment/RECOVERY.json')
    check('Isolated xLSTM environment optional-compiler patch documented',
          frontier_env['status'] == 'PASS_ISOLATED_IMPORT_ONLY_PATCH' and frontier_env['base_packages_unchanged']
          and not frontier_env['mLSTM_forward_training_math_changed']
          and frontier_env['exact_changed_paths'] == ['blocks/slstm/src/cuda_init.py'], 'late_frontier/environment/RECOVERY.json')
    combo = read('EPS_PE_COMBINATION_RECEIPT.json')
    numerical = read('EPS_CHANNEL_RETRAINING_NUMERICAL_REVIEW.json')
    check('Fresh-training discrepancy preserved and deterministic profile bounded',
          numerical['status']=='PASS_TWO_MODEL_2019_DETERMINISTIC_RETRAINING_PROFILE'
          and numerical['initial_input_arrays_exact'] and numerical['initial_failure_preserved']
          and len(numerical['checks'])==2 and all(c['entire_checkpoint_bytes_exact'] for c in numerical['checks'])
          and not numerical['original_scores_replaced'], 'EPS_CHANNEL_RETRAINING_NUMERICAL_REVIEW.json')
    channel_api = read('EPS_ONLY_CHANNEL_PUBLIC_API_AUDIT.json')
    check('Two-channel public API saved replay with explicit fresh-fit limitation',
          channel_api['status']=='PASS_SAVED_API_WITH_FRESH_TRAINING_NUMERICAL_LIMITATION'
          and len(channel_api['checks'])==10
          and all(c['public_predict_save_load_exact'] and c['original_saved_public_predict_matches_frozen'] for c in channel_api['checks']),
          'EPS_ONLY_CHANNEL_PUBLIC_API_AUDIT.json')
    channels = read('EPS_ONLY_CHANNEL_REVIEW.json')
    check('Two fixed EPS-TTM-only channel ablations scored and replayed separately',
          channels['status']=='PASS_TWO_FIXED_ABLATIONS_FRESH_REPLAY' and channels['trained_annual_models']==16
          and channels['fresh_replay_checks']==80 and channels['main_artifacts_unchanged']
          and channels['new_portfolio_members']==0, 'EPS_ONLY_CHANNEL_REVIEW.json')
    few_years = read('EPS_FEW_YEAR_UNCERTAINTY_REVIEW.json')
    check('Few-year statistical resolution and unverified assumptions disclosed',
          few_years['status']=='EXACT_DIAGNOSTIC_ENUMERATION_COMPLETE' and len(few_years['comparisons'])==4
          and not few_years['formal_p_values_claimed'] and not few_years['main_portfolio_changed'],
          'EPS_FEW_YEAR_UNCERTAINTY_REVIEW.json')
    check('PE partial result reported without strict certification', combo['strict_certified_valid'] == 0 and combo['status'] == 'PARTIAL_V04_REAL_SCENARIOS_C4_CONTRACT_BLOCKED', 'EPS_PE_COMBINATION_RECEIPT.json')
    envs = list((RUN/'environments').glob('*.json'))
    check('Seven isolated environment locks', len(envs) == 7, 'ENVIRONMENT_MANIFEST.json')
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'handoff_ready': all(c['pass'] for c in checks),
              'checks': checks, 'qualified_replay_identities': len(qualified), 'recorded_replay_identities': len(replay_models),
              'formal_certified': False, 'production_promoted': False,
              'known_limitations_not_resolved_by_this_handoff_gate': [
                  'Confirmed raw share-count unit defect: 17 base models and 8 frozen ensembles exposed; no clean-accounting-feature or deployment claim',
                  'Mixed YTD/TTM/standalone-duration financial features: 26 main bases, 8 ensembles, 2 late multivariate models and no-share ablations exposed; shadow selector is not applied to V1.3',
                  'Saved model replay is not universal exact retraining. Two-channel TimeXer fresh-fit numerical drift was retained; a separate two-model/2019 deterministic profile matched checkpoint bytes, not all years/hardware.',
                  'Current-vintage SEC/share-basis/cohort survivorship and pretrained-corpus overlap',
                  'Original101score retirement; corrected run is not a fresh heldout',
                  'Five diagnostic-only original outputs excluded from final ensemble pool',
                  'Original numerical batch failures preserved; separate strictFP32 diagnostics do not replace original scores',
                  'TabPFN singleton-size numerics unresolved; FP64 diagnostic failed and OOMed, no further retry',
                  'C4 real-input connection blocked; v04 static scenarios are not future-price certification',
                  'Correlated candidate selection, few independent years, exploratory subgroup/cluster intervals',
                  'Compact handoff excludes large fitted weights and raw SEC caches; not a deployable full-weight release']}
    save_json(RUN/'FINAL_VERIFICATION_SUMMARY.json', result)
    print('FINAL_HANDOFF_VERIFICATION', result['handoff_ready'], 'checks', len(checks), 'failures', [c['check'] for c in checks if not c['pass']], flush=True)
    if not result['handoff_ready']: raise RuntimeError('Final verification incomplete or failed')


if __name__ == '__main__': run()
