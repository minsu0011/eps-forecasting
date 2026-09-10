from research.eps_model_lab_v1.final_reports import smoke_pass


def test_explicit_single_smoke():
    assert smoke_pass({'status': 'PASS'})
    assert not smoke_pass({'status': 'FULL_RESEARCH_SCORED'})
    assert not smoke_pass({})


def test_explicit_multihead_smoke():
    assert smoke_pass([{'status': 'PASS'}, {'status': 'PASS'}])
    assert not smoke_pass([{'status': 'PASS'}, {'status': 'FAIL'}])
    assert not smoke_pass([])
    assert not smoke_pass([None])
