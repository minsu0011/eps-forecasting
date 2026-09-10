"""Establish V2 authority, immutable prescore protocol and preservation baseline."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import sys
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from research.eps_model_lab_v2.common import RUN, V1, LAB, utcnow, sha, save_json, read_json


def run():
    RUN.mkdir(exist_ok=False)
    for folder in ['data','data/source_filings','audit','logs','prescore','models','predictions','failures']:
        (RUN/folder).mkdir(parents=True)
    save_json(RUN/'RUN_STATE.json', {'created_utc':utcnow(), 'status':'V1_EVIDENCE_AND_SOURCE_AUTHORITY',
        'v1_output_readonly':str(V1), 'PE_readonly':True, 'formal_certified':False,
        'next_actions':['Authoritative source contexts and quarantine','Clean data rebuild and full prefix mutation audit',
            'Freeze clean data and pre-2019 folds before baseline and bounded model training'],
        'no_reuse_of_previous_ten_hour_deadline':True})
    # Complete directory metadata; content hashes for data, declarations and all
    # reasonably-sized files. Large fitted V1 binaries are never consumed/refit.
    paths = sorted(p for p in V1.rglob('*') if p.is_file())
    def fingerprint(path):
        st = path.stat()
        important = 'data' in path.relative_to(V1).parts or st.st_size <= 4*1024*1024
        return {'path':str(path), 'bytes':st.st_size, 'mtime_ns':st.st_mtime_ns,
                'sha256':sha(path) if important else None}
    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(fingerprint, paths))
    pe = read_json(V1/'FINAL_PE_READONLY_AUDIT.json')
    pe_paths = sorted({r['path'] for r in pe['records']})
    pe_records = []
    for p in pe_paths:
        path = Path(p)
        if not path.is_absolute():
            path = PROJECT/path
        pe_records.append({'path':str(path), 'sha256':sha(path)})
    save_json(RUN/'audit/READONLY_BASELINE.json', {'created_utc':utcnow(), 'v1':records,'PE':pe_records,
        'v1_content_hash_count':sum(r['sha256'] is not None for r in records),
        'large_v1_weights_protection':'size and mtime inventory; never opened for training; not full cryptographic rehash'}, immutable=True)
    save_json(RUN/'prescore/RESEARCH_PROTOCOL_V2.json', {
        'created_utc':utcnow(), 'data_identity':'EPS_DATA_IDENTITY_V2_CLEAN',
        'pre_2019_development_fold_years':[2015,2016,2017,2018],
        'development_origin_and_label_cutoff':'2019-01-01T00:00:00Z',
        'confirmation_years':[2019,2020,2021], 'confirmation_label_cutoff':'2022-01-01T00:00:00Z',
        'confirmation_name':'V2_RESEARCH_CONFIRMATION_ALREADY_VIEWED_NOT_FORMAL',
        'monitor_years':[2022,2023,2024,2025,2026], 'monitor_use':'DESCRIPTIVE_ONLY_NO_SELECTION',
        'annual_training':'Origin AND each target publication strictly before annual Jan 1 cutoff',
        'modes':{'expanding':'All available training history', 'rolling':'Most recent five origin years',
            'recency_weighted':'Expanding with two-year half-life; unsupported native models use explicit weighted sampling only'},
        'max_meaningful_variants_per_family':5, 'seed':1729, 'best_seed_selection':False,
        'transforms':['signed_raw','train_only_robust_asinh','baseline_residual_or_delta'],
        'primary_selection_tracks':['N','A'], 'market_track_primary':False,
        'native_targets':{'A':'next directly reported GAAP diluted EPS', 'B':'h1,h2,h3,h4 native direct EPS path',
            'C':'future native-ledger TTM EPS at Q+4 with direct/approximation method flags'},
        'formal_certified':False,
        'ensemble_selection':'All members and weights from development only; freeze BEFORE any confirmation/monitor result',
        'ensemble_family_cap':1, 'ensemble_max_members':8, 'ensemble_methods':['mean','median','trimmed_mean','nonnegative_stack'],
        'research_survivor_gates':{
            'all':{'coverage_min':1.0,'prediction_finite_required':True,'saved_replay_pass':True,
                'fresh_training_repeats_min':2,'PIT_mutation_violations_max':0,'unverified_accounting_consumption_max':0,
                'tail_p99_ratio_max':1.25,'worst_year_MAE_ratio_max':1.30,
                'negative_and_transition_MAE_ratio_max':1.50,'subgroup_gate_min_rows':20},
            'A':{'development_MAE_gain_min':0.03,'confirmation_MAE_gain_min':0.03,'MedianAE_ratio_max':1.0},
            'C':{'development_MAE_gain_min':0.03,'confirmation_MAE_gain_min':0.03,'MedianAE_ratio_max':1.0},
            'failure':'No gate relaxation; retain explicit LANE_C_NOT_READY if none pass',
            'baseline':'persistence_observed on identical finite-truth rows',
            'few_year_warning':'No exaggerated p-values; minimum subgroup counts and empty gates remain explicit'},
        'source_authority':'Exact filing context/unitRef/scale first; same-accession statement evidence next; unknown remains missing/quarantined',
        'hardware':{'cpu_benchmark_workers':[16,20,24,28],'inner_threads':1,'heavy_GPU_train_max':1,
            'steady_VRAM_GiB_max':15,'steady_system_RAM_GiB_max':80}}, immutable=True)
    print('V2_INITIALIZED', RUN, 'V1 files',len(records),'PE files',len(pe_records), flush=True)


if __name__ == '__main__':
    run()
