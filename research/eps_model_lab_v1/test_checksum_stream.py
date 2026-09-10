import hashlib
from research.eps_model_lab_v1.bootstrap import sha


def test_streaming_sha_matches_standard_for_multiple_chunks(tmp_path):
    data=bytes(range(256))*12345
    path=tmp_path/'payload.bin';path.write_bytes(data)
    assert sha(path)==hashlib.sha256(data).hexdigest()


def test_empty_file_sha(tmp_path):
    path=tmp_path/'empty';path.write_bytes(b'')
    assert sha(path)==hashlib.sha256(b'').hexdigest()
