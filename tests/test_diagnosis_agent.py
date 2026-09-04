import json

import pytest

from agents.http_agent import Diagnosis


class _SuccessfulResponse:
    status_code = 200
    text = ''
    headers = {'Content-Type': 'application/json'}

    def json(self):
        return {'content': {'data': {'markdown': 'ok'}}}

    def close(self):
        pass


class _SseResponse:
    status_code = 200
    headers = {'Content-Type': 'text/event-stream; charset=utf-8'}
    _lines = [
        'data: {"event_type":"start","data":{"status":"SUCCESS"}}',
        '',
        'data: {"event_type":"delta","data":{"contentDelta":"partial"}}',
        '',
        'data: {"event_type":"complete","data":{"status":"SUCCESS",'
        '"content":"最终诊断结果"}}',
        '',
        'data: {"event_type":"done","data":{"status":"SUCCESS"}}',
        '',
    ]
    text = '\n'.join(_lines)

    def json(self):
        return json.loads(self.text)

    def iter_lines(self, decode_unicode=False):
        return iter(self._lines)

    def close(self):
        pass


def test_diagnosis_extracts_complete_content_from_sse(monkeypatch):
    captured = {}

    def fake_post(url, data, headers, **kwargs):
        captured.update({'data': data, 'headers': headers, 'kwargs': kwargs})
        return _SseResponse()

    monkeypatch.setattr('agents.http_agent.requests.post', fake_post)
    agent = Diagnosis('https://example.test/chat-messages', 'token')

    response_raw, summary, _ = agent.call_agent('故障码122', language='zh-CN')

    assert response_raw['event_type'] == 'complete'
    assert summary == '最终诊断结果'
    assert json.loads(captured['data'])['response_mode'] == 'streaming'
    assert captured['kwargs']['stream'] is True


@pytest.mark.parametrize('language', ['zh-CN', 'en-US'])
def test_diagnosis_adds_language_request_header(monkeypatch, language):
    captured = {}

    def fake_post(url, data, headers, **kwargs):
        captured.update({
            'url': url,
            'data': data,
            'headers': headers,
            'kwargs': kwargs,
        })
        return _SuccessfulResponse()

    monkeypatch.setattr('agents.http_agent.requests.post', fake_post)
    agent = Diagnosis('https://example.test/chat-messages', 'token')

    _, summary, _ = agent.call_agent('故障码122', language=language)

    assert summary == 'ok'
    assert captured['headers']['Language'] == language
    assert captured['headers']['Accept'] == 'text/event-stream'
    assert captured['kwargs']['stream'] is True


def test_diagnosis_defaults_language_to_zh_cn(monkeypatch):
    captured_headers = {}

    def fake_post(url, data, headers, **kwargs):
        captured_headers.update(headers)
        return _SuccessfulResponse()

    monkeypatch.setattr('agents.http_agent.requests.post', fake_post)
    Diagnosis('https://example.test/chat-messages', 'token').call_agent('故障码122')

    assert captured_headers['Language'] == 'zh-CN'


def test_diagnosis_rejects_unsupported_language(monkeypatch):
    called = False

    def fake_post(url, data, headers, **kwargs):
        nonlocal called
        called = True
        return _SuccessfulResponse()

    monkeypatch.setattr('agents.http_agent.requests.post', fake_post)
    agent = Diagnosis('https://example.test/chat-messages', 'token')

    with pytest.raises(ValueError, match='仅支持'):
        agent.call_agent('fault code 122', language='en-GB')

    assert called is False
