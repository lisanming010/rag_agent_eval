import pytest

import main as evaluation_main
from agents.http_agent import Diagnosis


class _TestConfig:
    def get(self, key, default=None):
        if key == 'agents.http_agent.call_agent_max_retries':
            return 0
        if key == 'agents.http_agent.call_agent_retry_delay':
            return 0
        return default


class _CapturingAgent:
    def __init__(self):
        self.calls = []

    def call_agent(self, question, **kwargs):
        self.calls.append((question, kwargs))
        return {}, 'ok', '0.0001'


class _FailingAgent:
    def call_agent(self, question, **kwargs):
        raise RuntimeError('request failed')


class _RetryConfig:
    def get(self, key, default=None):
        if key == 'agents.http_agent.call_agent_max_retries':
            return 1
        if key == 'agents.http_agent.call_agent_retry_delay':
            return 0
        return default


class _SseResponse:
    status_code = 200
    headers = {'Content-Type': 'text/event-stream; charset=utf-8'}
    text = ''

    def iter_lines(self, decode_unicode=False):
        return iter([
            'data: {"event_type":"start","data":{"status":"SUCCESS"}}',
            'data: {"event_type":"complete","data":{"status":"SUCCESS",'
            '"content":"SSE 最终诊断内容"}}',
            'data: {"event_type":"done","data":{"status":"SUCCESS"}}',
        ])

    def close(self):
        pass


def test_single_turn_forwards_case_language(monkeypatch):
    monkeypatch.setattr(
        evaluation_main.ConfigReader,
        'get_instance',
        lambda: _TestConfig(),
    )
    agent = _CapturingAgent()
    case = {
        'query': 'fault code 122',
        'language': 'en-US',
        '_tenant_id': None,
    }

    evaluation_main.call_agent(agent, case)

    assert agent.calls == [(
        'fault code 122',
        {'tenant_id': None, 'language': 'en-US'},
    )]
    assert case['agent_response'] == 'ok'


def test_multi_turn_forwards_same_language_to_every_turn(monkeypatch):
    monkeypatch.setattr(
        evaluation_main.ConfigReader,
        'get_instance',
        lambda: _TestConfig(),
    )
    agent = _CapturingAgent()
    case = {
        '用例编号': 'TC-LANG-001',
        'language': 'zh-CN',
        '第1轮': '故障码122',
        '第1轮对话预期结果': '地址码重复故障',
        '第2轮': '如何处理？',
        '第2轮对话预期结果': '修改重复地址码',
        '_source_csv': 'test.csv',
        '_tenant_id': None,
    }

    rows = evaluation_main.call_multi_turn_agent(agent, case)

    assert len(rows) == 2
    assert [call[0] for call in agent.calls] == ['故障码122', '如何处理？']
    assert all(call[1]['language'] == 'zh-CN' for call in agent.calls)
    assert agent.calls[0][1]['session_id'] == agent.calls[1][1]['session_id']


def test_single_turn_writes_sse_complete_content_to_agent_response(monkeypatch):
    monkeypatch.setattr(
        evaluation_main.ConfigReader,
        'get_instance',
        lambda: _TestConfig(),
    )
    monkeypatch.setattr(
        'agents.http_agent.requests.post',
        lambda *args, **kwargs: _SseResponse(),
    )
    case = {
        'query': '故障码122',
        'language': 'zh-CN',
        '_tenant_id': None,
    }

    evaluation_main.call_agent(
        Diagnosis('https://example.test/chat-messages', 'token'),
        case,
    )

    assert case['agent_response'] == 'SSE 最终诊断内容'


def test_single_turn_emits_retry_and_reraises_final_failure(monkeypatch):
    events = []
    monkeypatch.setattr(
        evaluation_main.ConfigReader,
        'get_instance',
        lambda: _RetryConfig(),
    )
    monkeypatch.setattr(evaluation_main, 'emit_terminal_progress', events.append)

    with pytest.raises(RuntimeError, match='request failed'):
        evaluation_main.call_agent(
            _FailingAgent(),
            {'用例编号': 'TC-001', 'query': '故障码122'},
        )

    assert len(events) == 1
    assert events[0].event == 'retry'
    assert events[0].task_name == 'call_agent'
    assert '用例=TC-001' in events[0].message
    assert '尝试=2/2' in events[0].message
