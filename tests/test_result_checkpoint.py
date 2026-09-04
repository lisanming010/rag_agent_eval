"""提交故障注入：只操作 pytest 临时目录，不连接外部服务。"""

import json

import pytest

import tool.result_checkpoint as checkpoints
from tool.csv_writer import CsvWriter
from tool.result_checkpoint import CheckpointError, CheckpointWriter, checkpoint_path, read_committed


def result_path(tmp_path):
    return tmp_path / 'result_outputs_demo.csv'


def test_roundtrip_append_and_schema_expansion(tmp_path):
    path = result_path(tmp_path)
    writer = CheckpointWriter(path, 'Diagnosis')
    assert read_committed(path, 'Diagnosis').rows == []
    writer.commit([{'id': '001', 'agent_response': '中文,"quoted"\r\n下一行', 'is_success': True}])
    writer.commit([{'id': '002', 'agent_response': 'x' * 200000, 'is_success': False}])
    writer.commit([{'id': '003', 'extra_reason': 'new', 'agent_response': None}])
    rows = read_committed(path, 'Diagnosis').rows
    assert len(rows) == 3
    assert rows[0]['agent_response'] == '中文,"quoted"\r\n下一行'
    assert rows[0]['extra_reason'] == ''
    assert rows[2]['agent_response'] == ''
    state = json.loads(checkpoint_path(path).read_text(encoding='utf-8'))
    assert state['committed_rows'] == state['last_committed_batch'] == 3
    assert state['last_case_id'] == '003'
    assert path.read_bytes().count(b'\xef\xbb\xbf') == 1


@pytest.mark.parametrize('tail', [b'2,False\r\n', b'2,"unfinished', b'2,"\xe4\xb8'])
def test_partial_tail_is_not_parsed_or_reused(tmp_path, tail):
    path = result_path(tmp_path)
    writer = CheckpointWriter(path, 'Diagnosis')
    writer.commit([{'id': '1', 'is_success': True}])
    with path.open('ab') as target:
        target.write(tail)
    before = path.read_bytes()
    snapshot = read_committed(path, 'Diagnosis')
    assert snapshot.rows == [{'id': '1', 'is_success': 'True'}]
    assert snapshot.has_uncommitted_tail
    assert path.read_bytes() == before


def test_missing_pair_and_old_csv(tmp_path):
    path = result_path(tmp_path)
    assert read_committed(path, 'Diagnosis').rows == []
    CsvWriter(path).write_rows([{'id': '1'}])
    with pytest.raises(CheckpointError, match='缺少 checkpoint'):
        read_committed(path, 'Diagnosis')


@pytest.mark.parametrize('damage', ['digest', 'short', 'missing', 'json', 'agent', 'file', 'version', 'bool_count', 'columns'])
def test_corrupt_committed_results_fail_closed(tmp_path, damage):
    path = result_path(tmp_path)
    writer = CheckpointWriter(path, 'Diagnosis')
    writer.commit([{'id': '1', 'is_success': True}, {'id': '2', 'is_success': False}])
    marker = checkpoint_path(path)
    state = json.loads(marker.read_text(encoding='utf-8'))
    if damage == 'digest':
        path.write_bytes(path.read_bytes().replace(b'True', b'Fake'))
    elif damage == 'short':
        CsvWriter(path).write_rows([{'id': '1', 'is_success': True}])
    elif damage == 'missing':
        path.unlink()
    elif damage == 'json':
        marker.write_text('{', encoding='utf-8')
    else:
        field, value = {'agent': ('agent', 'dataqa'), 'file': ('result_file', 'other.csv'),
                        'version': ('version', 999), 'bool_count': ('committed_rows', True),
                        'columns': ('committed_columns', ['id', 'id'])}[damage]
        state[field] = value
        marker.write_text(json.dumps(state), encoding='utf-8')
    with pytest.raises(CheckpointError):
        read_committed(path, 'Diagnosis')


@pytest.mark.parametrize('expand_header', [False, True])
def test_checkpoint_failure_keeps_old_boundary(tmp_path, monkeypatch, expand_header):
    path = result_path(tmp_path)
    writer = CheckpointWriter(path, 'Diagnosis')
    writer.commit([{'id': '1', 'is_success': True}])
    marker_before = checkpoint_path(path).read_bytes()

    def fail(*args):
        raise OSError('checkpoint unavailable')

    monkeypatch.setattr(checkpoints, 'atomic_json', fail)
    row = {'id': '2', 'is_success': True}
    if expand_header:
        row['new_column'] = 'review result'
    with pytest.raises(OSError):
        writer.commit([row])
    assert checkpoint_path(path).read_bytes() == marker_before
    snapshot = read_committed(path, 'Diagnosis')
    assert snapshot.rows == [{'id': '1', 'is_success': 'True'}]
    assert snapshot.has_uncommitted_tail
    with pytest.raises(RuntimeError, match='禁止继续'):
        writer.commit([row])


def test_first_batch_failure_has_zero_checkpoint(tmp_path, monkeypatch):
    path = result_path(tmp_path)
    writer = CheckpointWriter(path, 'Diagnosis')

    def partial(self, rows):
        self.csv_path.write_bytes(b'\xef\xbb\xbfid\r\n"unfinished')
        raise OSError('disk full')

    monkeypatch.setattr(CsvWriter, 'write_rows', partial)
    with pytest.raises(OSError):
        writer.commit([{'id': '1'}])
    assert read_committed(path, 'Diagnosis').rows == []
    assert json.loads(checkpoint_path(path).read_text(encoding='utf-8'))['committed_rows'] == 0


def test_stable_schema_does_not_reread_prefix(tmp_path, monkeypatch):
    path = result_path(tmp_path)
    writer = CheckpointWriter(path, 'Diagnosis')
    writer.commit([{'id': '1'}])
    monkeypatch.setattr(checkpoints, 'read_csv_prefix', lambda *a: pytest.fail('不应重读整表'))
    writer.commit([{'id': '2'}])
    writer.commit([{'id': '3'}])


def test_never_overwrite_an_existing_result(tmp_path):
    path = result_path(tmp_path)
    writer = CheckpointWriter(path, 'Diagnosis')
    writer.commit([{'id': '1'}])
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        CheckpointWriter(path, 'Diagnosis')
    assert path.read_bytes() == before


def test_atomic_marker_replace_failure_preserves_previous_json(tmp_path, monkeypatch):
    path = result_path(tmp_path)
    writer = CheckpointWriter(path, 'Diagnosis')
    writer.commit([{'id': '1'}])
    before = checkpoint_path(path).read_bytes()

    def fail_replace(*args):
        raise PermissionError('locked')

    monkeypatch.setattr(checkpoints.os, 'replace', fail_replace)
    with pytest.raises(PermissionError):
        writer.commit([{'id': '2'}])
    assert checkpoint_path(path).read_bytes() == before
    assert not list(tmp_path.glob('*.tmp'))
    assert len(read_committed(path, 'Diagnosis').rows) == 1


@pytest.mark.parametrize('which', ['csv', 'checkpoint'])
def test_directories_cannot_masquerade_as_result_files(tmp_path, which):
    path = result_path(tmp_path)
    target = path if which == 'csv' else checkpoint_path(path)
    target.mkdir()
    with pytest.raises(CheckpointError, match='不是文件'):
        read_committed(path, 'Diagnosis')


def test_decode_error_inside_committed_content_is_not_ignored(tmp_path):
    path = result_path(tmp_path)
    writer = CheckpointWriter(path, 'Diagnosis')
    writer.commit([{'query': 'hello'}])
    path.write_bytes(path.read_bytes().replace(b'hello', b'\xff'))
    with pytest.raises(CheckpointError):
        read_committed(path, 'Diagnosis')
