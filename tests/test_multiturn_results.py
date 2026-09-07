"""多轮评价的每轮状态、整体状态及实际 CSV 消费链路回归测试。"""

import csv
from copy import deepcopy

import pytest

import main as evaluation_main
from tool.collection_result import CollectionResult
from tool.get_bad_cases import extract_bad_cases


def make_turns(verdicts):
    return [
        {
            '用例编号': 'MT001',
            '_parent_case_id': 'MT001',
            '_turn': turn,
            '_total_turns': len(verdicts),
            '_is_multi_turn_sub': True,
            'query': f'question {turn}',
            'agent_response': f'answer {turn}',
            'expected_behavior': f'expected {turn}',
            'judge_is_success': passed,
            'judge_reason': f'reason {turn}',
            'is_success': passed,
            '用例是否通过': passed,
        }
        for turn, passed in enumerate(verdicts, start=1)
    ]


@pytest.mark.parametrize('verdicts', [(True, False), (False, True), (True, True)])
def test_aggregation_only_adds_parent_status_without_changing_turns(verdicts):
    rows = make_turns(verdicts)
    rows[-1]['evaluate_error'] = 'original error'
    before = deepcopy(rows)
    batch = {'csv': rows}

    evaluation_main._aggregate_multi_turn(batch)
    evaluation_main._aggregate_multi_turn(batch)  # 重复聚合也不污染每轮错误信息。

    for row, original in zip(rows, before):
        assert row['_parent_all_pass'] is all(verdicts)
        assert {key: value for key, value in row.items()
                if key != '_parent_all_pass'} == original


def test_aggregation_does_not_change_independent_single_turn():
    row = {'query': 'single', 'is_success': True, 'evaluate_error': 'own error'}
    before = dict(row)
    evaluation_main._aggregate_multi_turn({'csv': [row]})
    assert row == before


def test_merge_uses_parent_status_without_changing_turn_details():
    rows = make_turns((True, False))
    for row in rows:
        row['_parent_all_pass'] = False
    before = deepcopy(rows)

    merged = evaluation_main._merge_multi_turn_rows(rows, normalize_single_turn=True)[0]

    assert merged['is_success'] is False
    assert merged['用例是否通过'] is False
    assert merged['_parent_all_pass'] is False
    assert merged['第1轮对话是否通过'] is True
    assert merged['第2轮对话是否通过'] is False
    assert merged['judge_is_success(t1)'] is True
    assert merged['judge_is_success(t2)'] is False
    assert rows == before


def test_tmp_merge_does_not_invent_evaluation_results():
    rows = make_turns((True, True))
    for row in rows:
        for key in ('is_success', '用例是否通过', 'judge_is_success', 'judge_reason'):
            row.pop(key)

    merged = evaluation_main._merge_multi_turn_rows(rows)[0]

    assert 'is_success' not in merged
    assert '_parent_all_pass' not in merged
    assert merged['agent_response(t1)'] == 'answer 1'
    assert merged['agent_response(t2)'] == 'answer 2'


@pytest.mark.parametrize('verdicts,review_override', [
    ((True, False), None),
    ((False, True), None),
    ((True, True), None),
    ((True, False, True), None),
    ((True, False), True),
    ((True, True), False),
])
def test_pipeline_preserves_turns_and_reports_parent_result(
    tmp_path, monkeypatch, verdicts, review_override,
):
    output_dir = tmp_path / 'results'

    class Config:
        def get(self, key, default=None):
            return {'result.save_path': str(output_dir)}.get(key, default)

    pipeline = evaluation_main.EvaluationPipeline.__new__(evaluation_main.EvaluationPipeline)
    pipeline.conf = Config()
    monkeypatch.setattr(evaluation_main, 'mkdir_with_timestamp', lambda path: path)
    rows = make_turns(verdicts)
    # 初评只写指标字段，整体状态由真实的流水线汇总。
    for row in rows:
        row.pop('is_success')
        row.pop('用例是否通过')
    if not verdicts[-1] and review_override is None:
        rows[-1]['evaluate_error'] = 'original turn failure'

    def review(batch):
        if review_override is not None:
            # 确保整体聚合使用复核后的最终指标状态，而非初评状态。
            batch['csv'][-1]['judge_is_success'] = review_override

    async def fake_groups(groups, metrics, on_result, **kwargs):
        from evaluator.runner import recompute_overall_success
        for group in groups:
            batch = {'csv': group}
            review(batch)
            recompute_overall_success(batch)
            await on_result(group)

    monkeypatch.setattr(evaluation_main, 'evaluate_groups', fake_groups)
    case = {'csv': rows, 'case_name': 'test_cases_multiturn.csv', 'metrics': ['judge']}

    pipeline._evaluate({'Diagnosis': [case]})

    final_verdicts = list(verdicts)
    if review_override is not None:
        final_verdicts[-1] = review_override
    parent_pass = all(final_verdicts)
    assert [row['is_success'] for row in rows] == final_verdicts
    assert [row['用例是否通过'] for row in rows] == final_verdicts
    assert all(row['_parent_all_pass'] is parent_pass for row in rows)
    for index, row in enumerate(rows):
        if index == len(rows) - 1 and not verdicts[-1] and review_override is None:
            assert row['evaluate_error'] == 'original turn failure'
        else:
            assert 'evaluate_error' not in row

    result_dir = output_dir / 'diagnosis'
    result_path = result_dir / 'result_outputs_multiturn.csv'
    with result_path.open(encoding='utf-8-sig', newline='') as source:
        written = list(csv.DictReader(source))
    assert len(written) == 1
    merged = written[0]
    assert merged['is_success'] == str(parent_pass)
    assert merged['用例是否通过'] == str(parent_pass)
    assert merged['_parent_all_pass'] == str(parent_pass)
    for turn, passed in enumerate(final_verdicts, start=1):
        assert merged[f'第{turn}轮对话是否通过'] == str(passed)
        assert merged[f'judge_is_success(t{turn})'] == str(passed)

    # 实际下游消费者必须得到正确的整体结果，而不是第一轮结果。
    assert CollectionResult(str(result_path)).task_success_stats()['total'] == (
        100.0 if parent_pass else 0.0
    )
    assert extract_bad_cases(str(result_dir))['multiturn'] == (
        0 if parent_pass else 1, 1
    )
