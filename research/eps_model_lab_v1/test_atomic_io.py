import json
from unittest.mock import patch
from research.eps_model_lab_v1.bootstrap import atomic_replace,save_json


def test_transient_windows_lock_retries_without_nonatomic_fallback(tmp_path):
    with patch('research.eps_model_lab_v1.bootstrap.os.replace',side_effect=[PermissionError('transient'),None]) as replace:
        with patch('research.eps_model_lab_v1.bootstrap.time.sleep'):
            atomic_replace('temporary','destination')
    assert replace.call_count==2
    assert replace.call_args.args==('temporary','destination')


def test_json_replacement_complete_and_no_shared_temp_name(tmp_path):
    path=tmp_path/'state.json';save_json(path,{'value':1});save_json(path,{'value':2})
    assert json.loads(path.read_text(encoding='utf-8'))=={'value':2}
    assert len(list(tmp_path.iterdir()))==1
