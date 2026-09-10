"""OOF-only shortlist: core, robust/subgroup and actual equal-pair usefulness."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
import numpy as np
import pandas as pd
from research.eps_model_lab_v1.bootstrap import RUN, save_json, sha
from research.eps_model_lab_v1.ensemble_models import EXCLUDED, family


def run():
    score = pd.read_csv(RUN/'EPS_SELECTION_LEADERBOARD_V1.csv')
    score = score[score.portfolio_eligible & score.coverage.ge(.999) & ~score.eps_model_id.isin(EXCLUDED)
                  & ~score.eps_model_id.str.startswith('ens_') & score.MAE.notna()]
    lock = json.loads((RUN/'ENSEMBLE_FREEZE_V1.json').read_text(encoding='utf-8'))
    evidence = {}; notes = []; limit = 8
    def add(name, role, target, reason):
        if name not in evidence and len(evidence) >= limit: return
        record = evidence.setdefault(name, {'model_id': name, 'family': family(name), 'roles': [], 'evidence': []})
        if role not in record['roles']: record['roles'].append(role)
        record['evidence'].append({'target': target, **reason})
    for target in ['h1', 'ttm']:
        for track in ['local_causal', 'mixed_retrospective']:
            s = score[score.target_key == target]
            if track == 'local_causal': s = s[~s.checkpoint_pretraining_overlap_unresolved]
            leader = s.sort_values(['MAE', 'eps_model_id']).iloc[0]
            add(leader.eps_model_id, 'EPS_CORE', target, {'basis': 'Best full-coverage OOF raw MAE in track', 'track': track,
                'MAE': float(leader.MAE), 'MedianAE': float(leader.MedianAE), 'price_scaled_MAE': float(leader.price_scaled_MAE), 'rows': int(leader.predicted_rows)})
            plan = next(p for p in lock['plans'] if p['track'] == track and p['target'] == target)
            add(plan['member_ids'][0], 'EPS_ENSEMBLE_COMPONENT', target,
                {'basis': 'First member under frozen three-metric OOF ranking', 'track': track})
    # Robustness and subgroup roles require actual OOF support, not names alone.
    for target in ['h1', 'ttm']:
        s = score[score.target_key == target].copy()
        s['robust_rank'] = s[['MedianAE', 'price_scaled_MAE']].rank(pct=True).mean(axis=1)
        r = s.sort_values(['robust_rank', 'eps_model_id']).iloc[0]
        add(r.eps_model_id, 'EPS_ROBUST', target, {'basis': 'OOF MedianAE/price-scaled rank; not formal tail certification',
            'MedianAE': float(r.MedianAE), 'price_scaled_MAE': float(r.price_scaled_MAE), 'MAE': float(r.MAE)})
        for group, role in [('negative', 'EPS_NEGATIVE_EARNINGS_SPECIALIST'), ('high_growth', 'EPS_GROWTH_SPECIALIST')]:
            q = s[s[group+'_n'].ge(30) & s[group+'_MAE'].notna()]
            if q.empty:
                notes.append({'target': target, 'group': group, 'status': 'INSUFFICIENT_30_ROW_SUPPORT_NO_SPECIALIST_CLAIM'})
                continue
            r = q.sort_values([group+'_MAE', 'eps_model_id']).iloc[0]
            add(r.eps_model_id, role, target, {'basis': 'Exploratory best supported OOF subgroup MAE, not multiplicity-adjusted',
                'subgroup': group, 'subgroup_rows': int(r[group+'_n']), 'subgroup_MAE': float(r[group+'_MAE']), 'overall_MAE': float(r.MAE)})
    pairs = pd.read_csv(RUN/'EPS_COMPLEMENTARITY_OOF.csv'); pair_candidates = []
    for target in ['h1', 'ttm']:
        s = score[score.target_key == target].set_index('eps_model_id'); expected = int(s.predicted_rows.max())
        p = pairs[(pairs.target_key == target) & pairs.rows.eq(expected)]
        for row in p.itertuples():
            if row.model_a not in s.index or row.model_b not in s.index: continue
            best_single = min(float(s.loc[row.model_a, 'MAE']), float(s.loc[row.model_b, 'MAE']))
            gain = best_single-float(row.equal_pair_actual_MAE)
            if gain <= 0: continue
            if row.model_a not in evidence and row.model_b not in evidence: continue
            other = row.model_b if row.model_a in evidence else row.model_a
            pair_candidates.append({'model_id': other, 'target': target, 'model_a': row.model_a, 'model_b': row.model_b,
                'equal_pair_actual_MAE': float(row.equal_pair_actual_MAE), 'gain_over_better_single': gain,
                'relative_gain': gain/best_single, 'residual_corr': float(row.residual_corr), 'rows': int(row.rows),
                'basis': 'Actual equal-weight pair beats both singles on exact full OOF mask; not oracle score'})
    limit = 10
    for p in sorted(pair_candidates, key=lambda x: -x['relative_gain']):
        add(p['model_id'], 'EPS_DIVERSITY_COMPONENT', p['target'], {k:v for k,v in p.items() if k not in ['model_id', 'target']})
        if sum('EPS_DIVERSITY_COMPONENT' in r['roles'] for r in evidence.values()) >= 2: break
    for name, record in evidence.items():
        record['metrics_by_target'] = {}
        for _, row in score[score.eps_model_id == name].iterrows():
            record['metrics_by_target'][row.target_key] = {k: float(row[k]) for k in ['MAE', 'MedianAE', 'price_scaled_MAE', 'coverage']}
        record.update(formal_certified=False, deployable=False, selection_uses_research_test=False)
    save_json(RUN/'EPS_RESEARCH_SHORTLIST_V1.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
              'models': list(evidence.values()), 'notes': notes, 'selection_surface': 'SELECTION_OOF_AVAILABLE_BEFORE_2022',
              'selection_score_sha256': sha(RUN/'EPS_SELECTION_LEADERBOARD_V1.csv'),
              'ensemble_freeze_sha256': sha(RUN/'ENSEMBLE_FREEZE_V1.json'), 'roles_use_test': False,
              'caveats': ['Exploratory correlated candidates and small historical-year count',
                         'No cyclical-specialist claim without verified historical sector/regime support',
                         'Subgroup winner is a next-wave hypothesis, not a separately validated specialist model']})
    print('OOF_SHORTLIST_COMPLETE', len(evidence), flush=True)


if __name__ == '__main__': run()
