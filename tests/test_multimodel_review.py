"""离线验证复核调用失败及非布尔结果的投票行为。"""

import pytest

import evaluator.runner as runner


MISSING = object()
CALL_ERROR = object()
METRIC_NAME = 'Contextual Recall'


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
def test_review_requires_two_explicit_true_votes(model1, model2, model3, expected):
    case = {}
    for suffix, verdict in zip(('', '_model2', '_model3'), (model1, model2, model3)):
        if verdict is not MISSING and verdict is not CALL_ERROR:
            case[f'{METRIC_NAME}{suffix}_is_success'] = verdict
    runner._apply_voting(case, [METRIC_NAME])
    assert case[f'{METRIC_NAME}_is_success'] is expected
    assert case['is_success'] is expected
    assert case['用例是否通过'] is expected


@pytest.mark.parametrize('verdict', [True, False, None, 'True', 'False', 1, 0, ''])
def test_overall_success_requires_boolean_true(verdict):
    case = {'judge_is_success': verdict}

    runner.recompute_overall_success({'csv': [case]})

    assert case['is_success'] is (verdict is True)
    assert case['用例是否通过'] is (verdict is True)
