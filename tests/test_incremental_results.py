"""离线验证评价批次落盘，不调用 Agent 或真实评价模型。"""

import csv
import threading
from types import SimpleNamespace

import pytest

import main as evaluation_main
import evaluator.runner as evaluation_runner
import tool.csv_writer as csv_writer_module
from tool.async_result_writer import AsyncResultWriter
from tool.csv_writer import CsvWriter


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as source:
        return list(csv.DictReader(source))


def test_append_aligns_columns_and_expands_header(tmp_path):
    path = tmp_path / 'results.csv'
    writer = CsvWriter(path)
    writer.append_rows([{'query': '中文,\n问题', 'score': 1}])
    writer.append_rows([{'score': 0, 'query': 'second'}])
    writer.append_rows([{'query': 'third', 'retry_count': 2, 'model2_reason': '复核'}])
    writer.append_rows([{'query': 'fourth'}])
    rows = read_rows(path)
    assert len(rows) == 4
    assert rows[0] == {'query': '中文,\n问题', 'score': '1',
                       'retry_count': '', 'model2_reason': ''}
    assert rows[1]['score'] == '0'
    assert rows[2]['retry_count'] == '2'
    assert rows[2]['score'] == ''
    assert rows[3]['model2_reason'] == ''
    assert path.read_bytes().count(b'\xef\xbb\xbf') == 1


def test_stable_header_uses_append_without_replacing(tmp_path, monkeypatch):
    path = tmp_path / 'results.csv'
    writer = CsvWriter(path)
    writer.write_rows([{'query': 'first', 'score': 1}])

    def unexpected_replace(*args):
        pytest.fail('稳定表头的批次不应重写整个文件')

    monkeypatch.setattr(csv_writer_module.os, 'replace', unexpected_replace)
    writer.append_rows([{'query': 'next'}])
    assert len(read_rows(path)) == 2


def test_header_expansion_preserves_large_response(tmp_path):
    path = tmp_path / 'results.csv'
    response = 'response\n' * 20000
    writer = CsvWriter(path)
    writer.write_rows([{'query': 'first', 'agent_response': response}])
    writer.append_rows([{'query': 'next', 'evaluate_error': 'failed'}])
    assert read_rows(path)[0]['agent_response'] == response


def test_failed_header_replacement_preserves_original(tmp_path, monkeypatch):
    path = tmp_path / 'results.csv'
    writer = CsvWriter(path)
    writer.write_rows([{'query': 'first'}])
    before = path.read_bytes()

    def fail_replace(*args):
        raise PermissionError('file is open')

    monkeypatch.setattr(csv_writer_module.os, 'replace', fail_replace)
    with pytest.raises(PermissionError):
        writer.append_rows([{'query': 'second', 'new_column': 'new'}])
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_async_writer_snapshot_flush_and_shutdown(tmp_path, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    original_write = CsvWriter.write_rows

    def delayed_write(self, rows, *args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original_write(self, rows, *args, **kwargs)

    monkeypatch.setattr(CsvWriter, 'write_rows', delayed_write)
    writer = AsyncResultWriter(tmp_path)
    writer.start()
    case = {'csv': [{'query': 'original'}]}
    try:
        writer.submit(case, 'test_cases_demo.csv')
        assert entered.wait(5)
        case['csv'][0]['query'] = 'mutated'
        case['csv'].append({'query': 'not submitted'})
        release.set()
        writer.flush()
        assert read_rows(tmp_path / 'result_outputs_demo.csv') == [{'query': 'original'}]
        writer.submit({'csv': [{'query': 'second'}]}, 'test_cases_demo.csv', append=True)
        writer.flush()
    finally:
        release.set()
        writer.wait_and_stop()
    assert len(read_rows(tmp_path / 'result_outputs_demo.csv')) == 2
    assert not writer.writer_thread.is_alive()
    assert writer.result_queue.unfinished_tasks == 0
    writer.wait_and_stop()  # 可重复收尾，不会死锁。


def make_pipeline(tmp_path, monkeypatch, batch_size=None):
    values = {'result.save_path': str(tmp_path)}
    if batch_size is not None:
        values['result.write_batch_size'] = batch_size

    class Config:
        def get(self, key, default=None):
            return values.get(key, default)

    pipeline = evaluation_main.EvaluationPipeline.__new__(evaluation_main.EvaluationPipeline)
    pipeline.conf = Config()
    monkeypatch.setattr(evaluation_main, 'mkdir_with_timestamp', lambda path: path)
    return pipeline


def install_fake_evaluation(monkeypatch, before_evaluate=None):
    calls = []

    def evaluate(batch):
        if before_evaluate:
            before_evaluate(len(calls), batch)
        calls.append([row['query'] for row in batch['csv']])
        for row in batch['csv']:
            row['judge_score'] = 1
            row['judge_is_success'] = True
            row['is_success'] = True

    def structured(batch):
        for row in batch['csv']:
            row['structured_reason'] = 'structured complete'

    def review(batch):
        for row in batch['csv']:
            row['judge_model2_reason'] = 'review complete'

    monkeypatch.setattr(evaluation_main, 'run_evaluate', evaluate)
    monkeypatch.setattr(evaluation_main, 'run_evaluate_structured', structured)
    monkeypatch.setattr(evaluation_main, 'run_multimodel_reevaluate', review)
    return calls


def dataset(count):
    return {'case_name': 'test_cases_demo.csv', 'metrics': ['reverse_validation'],
            'csv': [{'query': f'q{i}'} for i in range(count)]}


@pytest.mark.parametrize('results,initial_success,expected_queries', [
    ([(True, True), (True, True)], None, []),
    ([(True, True), (False, True)], None, ['q1']),
    ([(True, True), (True, False)], None, ['q1']),
    ([(True, True), (False, True)], True, ['q1']),
    ([(True, True), (True, True)], False, []),
])
def test_pipeline_reviews_only_failed_initial_evaluations(
    tmp_path, monkeypatch, results, initial_success, expected_queries,
):
    pipeline = make_pipeline(tmp_path, monkeypatch)
    cases = dataset(len(results))
    cases['metrics'].append('dataqa_capability')
    for row in cases['csv']:
        row['llm_test_case'] = object()
        if initial_success is not None:
            row['is_success'] = initial_success

    # 与真实初评相同，只写指标结果，不写整体 is_success。
    def evaluate(batch):
        for row, (llm_pass, _) in zip(batch['csv'], results):
            row['judge_is_success'] = llm_pass

    def structured(batch):
        for row, (_, structured_pass) in zip(batch['csv'], results):
            row['DataQA_Capability_is_success'] = structured_pass

    values = {
        'judge_llm.anthropic.model2': 'reviewer2',
        'judge_llm.anthropic.model3': 'reviewer3',
    }

    class Config:
        def get(self, key, default=None):
            return values.get(key, default)

    monkeypatch.setattr(evaluation_runner.ConfigReader, 'get_instance', lambda: Config())
    monkeypatch.setattr(evaluation_runner, 'METRIC_NEEDS_REVIEW', {'reverse_validation'})
    monkeypatch.setattr(evaluation_runner, 'create_metrics_for_model',
                        lambda model, metrics: {'judge': SimpleNamespace(__name__='judge')})
    reviewed = []

    def review_batch(rows, metrics, label, *args):
        reviewed.append((label, [row['query'] for row in rows]))
        for row in rows:
            row[f'judge_{label}_is_success'] = True

    monkeypatch.setattr(evaluation_main, 'run_evaluate', evaluate)
    monkeypatch.setattr(evaluation_main, 'run_evaluate_structured', structured)
    monkeypatch.setattr(evaluation_runner, '_batch_reevaluate', review_batch)

    pipeline._evaluate({'Diagnosis': [cases]})

    assert reviewed == ([(label, expected_queries) for label in ('model2', 'model3')]
                        if expected_queries else [])


@pytest.mark.parametrize('batch_size,count,expected', [
    (None, 45, [20, 20, 5]), (10, 23, [10, 10, 3]),
    (30, 31, [30, 1]), (20, 3, [3]), (20, 40, [20, 20]),
])
def test_pipeline_persists_completed_batches_before_next_evaluation(
    tmp_path, monkeypatch, batch_size, count, expected,
):
    pipeline = make_pipeline(tmp_path, monkeypatch, batch_size)
    path = tmp_path / 'diagnosis' / 'result_outputs_demo.csv'

    def before(batch_index, batch):
        if batch_index:
            written = read_rows(path)
            assert len(written) == sum(expected[:batch_index])
            assert all(row['structured_reason'] == 'structured complete' for row in written)
            assert all(row['judge_model2_reason'] == 'review complete' for row in written)
            assert all(row['is_success'] == 'True' for row in written)
            # 模拟后续批次重试新增字段。
            batch['csv'][0]['retry_count'] = 1

    calls = install_fake_evaluation(monkeypatch, before)
    cases = dataset(count)
    pipeline._evaluate({'Diagnosis': [cases]})
    written = read_rows(path)
    assert list(map(len, calls)) == expected
    assert [row['query'] for row in written] == [f'q{i}' for i in range(count)]
    assert len(cases['csv']) == count
    if len(expected) > 1:
        assert written[0]['retry_count'] == ''
        assert written[expected[0]]['retry_count'] == '1'


def test_multiturn_groups_not_split_and_mixed_schema_kept(tmp_path, monkeypatch):
    pipeline = make_pipeline(tmp_path, monkeypatch, 10)
    cases = dataset(11)
    turns = [{'query': f'multi{i}', '_parent_case_id': 'parent', '_turn': i,
              '_total_turns': 2, '_is_multi_turn_sub': True} for i in (1, 2)]
    cases['csv'].insert(0, turns[1])
    cases['csv'].append(turns[0])
    calls = install_fake_evaluation(monkeypatch)
    pipeline._evaluate({'Diagnosis': [cases]})
    written = read_rows(tmp_path / 'diagnosis' / 'result_outputs_demo.csv')
    assert list(map(len, calls)) == [10, 3]
    assert len(written) == 12
    assert 'query' not in written[0]
    assert [row['query(t1)'] for row in written[:11]] == [f'q{i}' for i in range(11)]
    assert written[-1]['query(t1)'] == 'multi1'
    assert written[-1]['query(t2)'] == 'multi2'
    assert written[0]['query(t2)'] == ''


@pytest.mark.parametrize('error_type', [RuntimeError, KeyboardInterrupt])
def test_evaluation_interruption_keeps_completed_batch(tmp_path, monkeypatch, error_type):
    pipeline = make_pipeline(tmp_path, monkeypatch, 10)

    def before(index, batch):
        if index == 1:
            raise error_type('interrupted')

    install_fake_evaluation(monkeypatch, before)
    with pytest.raises(error_type, match='interrupted'):
        pipeline._evaluate({'Diagnosis': [dataset(25)]})
    assert len(read_rows(tmp_path / 'diagnosis' / 'result_outputs_demo.csv')) == 10
    assert not any(t.name == 'AsyncResultWriter' for t in threading.enumerate())


def test_disk_failure_stops_before_next_evaluation(tmp_path, monkeypatch):
    pipeline = make_pipeline(tmp_path, monkeypatch, 10)
    calls = install_fake_evaluation(monkeypatch)

    def fail_append(self, data):
        raise OSError('disk full')

    monkeypatch.setattr(CsvWriter, 'append_rows', fail_append)
    with pytest.raises(RuntimeError, match='disk full'):
        pipeline._evaluate({'Diagnosis': [dataset(25)]})
    assert list(map(len, calls)) == [10, 10]
    assert len(read_rows(tmp_path / 'diagnosis' / 'result_outputs_demo.csv')) == 10
    assert not any(t.name == 'AsyncResultWriter' for t in threading.enumerate())


@pytest.mark.parametrize('batch_size', [0, 9, 31, True, '20', 20.5])
def test_invalid_batch_size_fails_before_evaluation(tmp_path, monkeypatch, batch_size):
    pipeline = make_pipeline(tmp_path, monkeypatch, batch_size)
    calls = install_fake_evaluation(monkeypatch)
    with pytest.raises(ValueError, match='write_batch_size'):
        pipeline._evaluate({'Diagnosis': [dataset(1)]})
    assert calls == []


def test_separate_datasets_and_classes_keep_separate_output(tmp_path, monkeypatch):
    pipeline = make_pipeline(tmp_path, monkeypatch, 10)
    install_fake_evaluation(monkeypatch)
    en = dataset(12)
    zh = dataset(3)
    zh['case_name'] = 'test_cases_zh.csv'
    pipeline._evaluate({'Diagnosis': [en, zh], '__shared__': [dataset(1)]})
    assert len(read_rows(tmp_path / 'diagnosis' / 'result_outputs_demo.csv')) == 12
    assert len(read_rows(tmp_path / 'diagnosis' / 'result_outputs_zh.csv')) == 3
    assert len(read_rows(tmp_path / 'default' / 'result_outputs_demo.csv')) == 1
