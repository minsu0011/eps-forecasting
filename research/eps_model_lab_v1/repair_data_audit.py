"""Freeze repair lineage and quantify data changes without consulting model scores."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
from research.eps_model_lab_v1.bootstrap import RUN,ORIGINAL_RUN,save_json,sha
from research.eps_model_lab_v1.common import features


def run():
    new=pd.read_parquet(RUN/'data/samples.parquet').set_index('sample_id')
    old=pd.read_parquet(ORIGINAL_RUN/'data/samples.parquet').set_index('sample_id')
    common=old.index.intersection(new.index);a=old.loc[common];b=new.loc[common]
    fa=features(a);fb=features(b)
    same=(fa.eq(fb)|(fa.isna()&fb.isna())).all(axis=1)
    labelcols=['y_h1','y_h2','y_h3','y_h4','y_ttm']
    labelsame=(a[labelcols].eq(b[labelcols])|(a[labelcols].isna()&b[labelcols].isna())).all(axis=1)
    records=[]
    for ticker in sorted(set(new.ticker)|set(old.ticker)):
        mask=a.ticker.eq(ticker)
        records.append({'ticker':ticker,'old_origins':int(old.ticker.eq(ticker).sum()),'new_origins':int(new.ticker.eq(ticker).sum()),
          'common_origins':int(mask.sum()),'changed_feature_origins':int((mask&~same).sum()),'changed_label_origins':int((mask&~labelsame).sum()),
          'added_sample_ids':list(new.index[new.ticker.eq(ticker)].difference(old.index)),
          'removed_sample_ids':list(old.index[old.ticker.eq(ticker)].difference(new.index))})
    lineage=json.loads((RUN/'PIT_REPAIR_LINEAGE.json').read_text(encoding='utf-8'))
    lineage['initial_staging_fix']=lineage['fix']
    lineage.update(fix='V1.3 native EPS/NI/CFO period-context fiscal calendar with first-available frozen anchor and fail-closed collisions',
       final_geometry='EPS_DATA_GEOMETRY_LOCK_V1_3.json',final_geometry_sha256=sha(RUN/'EPS_DATA_GEOMETRY_LOCK_V1_3.json'),
       final_samples_sha256=sha(RUN/'data/samples.parquet'),final_panel_sha256=sha(RUN/'data/fiscal_panel.parquet'),
       updated_utc=datetime.now(timezone.utc).isoformat(),same_base_model_recipes=True,
       ensemble_family_classification_correction='Darts classified CLASSICAL_STATISTICAL before new OOF selection')
    save_json(RUN/'PIT_REPAIR_LINEAGE.json',lineage)
    save_json(RUN/'PIT_REPAIR_DATA_DIFF.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
       'old_samples_sha256':sha(ORIGINAL_RUN/'data/samples.parquet'),'new_samples_sha256':sha(RUN/'data/samples.parquet'),
       'old_origins':len(old),'new_origins':len(new),'common_origins':len(common),
       'changed_feature_origins':int((~same).sum()),'changed_label_origins':int((~labelsame).sum()),
       'selection_basis':'Data contracts/source integrity only; no model scores consulted',
       'normalized_fiscal_year_labels_not_compared_as_features':True,'companies':records})
    pd.DataFrame(records).to_csv(RUN/'PIT_REPAIR_DATA_DIFF.csv',index=False)
    print('REPAIR_DIFF',len(old),len(new),'FEATURE_CHANGED',int((~same).sum()),'LABEL_CHANGED',int((~labelsame).sum()),flush=True)


if __name__=='__main__':run()
