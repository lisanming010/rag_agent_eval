"""离线验证复核调用失败及非布尔结果的投票行为。"""

from types import SimpleNamespace

import pytest

import evaluator.runner as runner


MISSING = object()
CALL_ERROR = object()
METRIC_NAME = 'Contextual Recall'


@pytest.fixture
def review_config(monkeypatch):
    values = {
        'judge_llm.anthropic.model2': 'reviewer2',
        'judge_llm.anthropic.model3': 'reviewer3',
        'evluate.run_async': False,
        'evluate.max_concurrent': 1,
        'evluate.throttle_value': 0,
        'retry.eval_max_retries': 0,
    }

    class Config:
        def get(self, key, default=None):
            return values.get(key, default)

    monkeypatch.setattr(runner.ConfigReader, 'get_instance', lambda: Config())
    monkeypatch.setattr(runner, 'METRIC_NEEDS_REVIEW', {'contextual_recall'})
    monkeypatch.setattr(
        runner, 'create_metrics_for_model',
        lambda model, names: {'contextual_recall': SimpleNamespace(__name__=METRIC_NAME)},
    )
    return values


@pytest.mark.parametrize('model1,model2,model3,expected', [
    pytest.param(False, True, True, True, id='two-explicit-passes'),
    pytest.param(False, True, False, False, id='two-explicit-failures'),
    pytest.param(False, True, MISSING, False, id='model3-missing'),
    pytest.param(True, MISSING, False, False, id='model2-missing'),
    pytest.param(True, MISSING, MISSING, False, id='both-reviews-missing'),
    pytest.param(MISSING, MISSING, MISSING, False, id='all-results-missing'),
    pytest.param(MISSING, True, True, True, id='missing-primary-two-passes'),
    pytest.param(False, True, CALL_ERROR, False, id='model3-call-failed'),
    pytest.param(True, CALL_ERROR, False, False, id='model2-call-failed'),
    pytest.param(True, CALL_ERROR, CALL_ERROR, False, id='both-calls-failed'),
    pytest.param(False, True, None, False, id='null-result'),
    pytest.param(False, True, 'True', False, id='string-true-is-not-boolean'),
    pytest.param(False, 'False', True, False, id='string-false-is-not-boolean'),
    pytest.param(False, True, 1, False, id='one-is-not-boolean'),
    pytest.param('True', True, False, False, id='primary-string-is-not-boolean'),
    pytest.param(1, True, False, False, id='primary-one-is-not-boolean'),
])
def test_review_requires_two_explicit_true_votes(
    monkeypatch, review_config, model1, model2, model3, expected,
):
    case = {'query': 'question', 'llm_test_case': SimpleNamespace(input='question'),
            'is_success': False}
    if model1 is not MISSING:
        case[f'{METRIC_NAME}_is_success'] = model1
    results = iter((model2, model3))
    calls = []

    def evaluate(cases, metrics, **kwargs):
        calls.append([c.input for c in cases])
        verdict = next(results)
        if verdict is CALL_ERROR:
            raise RuntimeError('model unavailable')
        if verdict is MISSING:
            return SimpleNamespace(test_results=[])
        metric = SimpleNamespace(name=METRIC_NAME, success=verdict,
                                 score=None if verdict is None else 0.5,
                                 threshold=0.7, reason='test verdict')
        return SimpleNamespace(test_results=[
            SimpleNamespace(input='question', metrics_data=[metric]),
        ])

    monkeypatch.setattr(runner, 'evaluate', evaluate)
    batch = {'csv': [case], 'metrics': ['contextual_recall']}

    runner.run_multimodel_reevaluate(batch)
    runner.recompute_overall_success(batch)

    assert calls == [['question'], ['question']]
    assert case.get(f'{METRIC_NAME}_is_success') is expected
    assert case['is_success'] is expected
    assert case['用例是否通过'] is expected


@pytest.mark.parametrize('verdict', [True, False, None, 'True', 'False', 1, 0, ''])
def test_overall_success_requires_boolean_true(verdict):
    case = {'judge_is_success': verdict}

    runner.recompute_overall_success({'csv': [case]})

    assert case['is_success'] is (verdict is True)
    assert case['用例是否通过'] is (verdict is True)
