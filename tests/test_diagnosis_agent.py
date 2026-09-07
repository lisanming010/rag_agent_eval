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


def _response_with_content(content, mode='blocking'):
    if mode == 'blocking':
        response = _SuccessfulResponse()
        response.json = lambda: {'content': {'data': {'markdown': content}}}
    else:
        response = _SseResponse()
        # 拆开标签，覆盖仅有 delta 时拼接完整响应后再解析的路径。
        response._lines = [
            'data: ' + json.dumps({
                'event_type': 'delta', 'data': {'contentDelta': part},
            })
            for part in (content[:4], content[4:])
        ]
    return response


@pytest.mark.parametrize('mode', ['blocking', 'streaming'])
@pytest.mark.parametrize('model', ['SG40CX', 'SG40CX-P2', 'SG40CX-p2'])
@pytest.mark.parametrize('session_id', [None, 'existing-session'])
def test_diagnosis_selects_exact_model_and_reuses_session(monkeypatch, mode, model, session_id):
    options = [
        f'{i}. MPPT Reverse Connection (Device (Inverter), Manufacturer (Sungrow), Model ({value}))'
        for i, value in enumerate(['SG40CX-P2', 'SG40CX-p2', 'SG40CX'], 1)
    ]
    first = '\n&#x20;\n'.join(f'<suggest>{option}</suggest>' for option in options)
    responses = iter([_response_with_content(first, mode), _response_with_content('最终诊断', mode)])
    requests = []

    def fake_post(url, data, headers, **kwargs):
        requests.append((json.loads(data), headers, kwargs))
        return next(responses)

    monkeypatch.setattr('agents.http_agent.requests.post', fake_post)
    agent = Diagnosis('https://example.test/chat-messages', 'token')
    query = f'Sungrow inverter reported fault code 265, model: {model}'
    _, summary, elapsed = agent.call_agent(
        query, user='test-user', session_id=session_id, language='en-US', res_mode=mode,
    )

    assert summary == '最终诊断'
    assert float(elapsed) >= 0
    assert len(requests) == 2
    assert requests[0][0]['query'] == query
    selected = options[['SG40CX-P2', 'SG40CX-p2', 'SG40CX'].index(model)]
    assert requests[1][0] == {**requests[0][0], 'query': selected}
    assert requests[0][1] == requests[1][1]
    assert requests[0][1]['X-Session-Id']
    if session_id is not None:
        assert requests[1][1]['X-Session-Id'] == session_id
    assert requests[1][1]['Language'] == 'en-US'
    assert requests[1][0]['user'] == 'test-user'
    assert requests[1][0]['response_mode'] == mode


@pytest.mark.parametrize(('query', 'content'), [
    ('model: SG40CX', '<suggest>1. Model (SG40CX-P2)</suggest>'),
    ('model: SG40CX-P2', '<suggest>1. Model (SG40CX)</suggest>'),
    ('model: SG40CX-p2', '<suggest>1. Model (SG40CX-P2)</suggest>'),
    ('fault code 265', '<suggest>1. Model (SG40CX)</suggest>'),
    ('model: ', '<suggest>1. Model (SG40CX)</suggest>'),
    ('model: SG40CX', '<suggest>1. SG40CX</suggest>'),
    ('model: SG40CX', '<suggest>1. Model (SG40CX)'),
    ('model: SG40CX', 'Model (SG40CX)'),
    ('model: SG40CX', '<suggest>1. Model (SG40CX)</suggest><suggest>2. Model (SG40CX)</suggest>'),
    ('model: SG40CX, model: SG40CX-P2', '<suggest>1. Model (SG40CX)</suggest>'),
])
def test_diagnosis_keeps_first_response_without_unique_match(monkeypatch, query, content):
    calls = []

    def fake_post(*args, **kwargs):
        calls.append(kwargs)
        return _response_with_content(content)

    monkeypatch.setattr('agents.http_agent.requests.post', fake_post)
    raw, summary, _ = Diagnosis('https://example.test/chat-messages', 'token').call_agent(query)
    assert summary == content
    assert raw['content']['data']['markdown'] == content
    assert len(calls) == 1


@pytest.mark.parametrize('query', ['故障码265，型号是 SG40CX ', '型号：SG40CX', 'model: SG40CX , fault: 265'])
def test_diagnosis_submits_suggest_text_only_once(monkeypatch, query):
    content = '<suggest>1. 故障（型号（SG40CX））</suggest>'
    calls = []

    def fake_post(url, data, **kwargs):
        calls.append(json.loads(data)['query'])
        return _response_with_content(content)

    monkeypatch.setattr('agents.http_agent.requests.post', fake_post)
    _, summary, _ = Diagnosis('https://example.test/chat-messages', 'token').call_agent(query)
    assert calls == [query, '1. 故障（型号（SG40CX））']
    assert summary == content


def test_diagnosis_propagates_second_request_failure(monkeypatch):
    calls = []

    def fake_post(*args, **kwargs):
        calls.append(kwargs)
        response = _response_with_content('<suggest>1. Model (SG40CX)</suggest>')
        if len(calls) == 2:
            response.status_code = 500
            response.text = 'second request failed'
        return response

    monkeypatch.setattr('agents.http_agent.requests.post', fake_post)
    with pytest.raises(RuntimeError, match='second request failed'):
        Diagnosis('https://example.test/chat-messages', 'token').call_agent('model: SG40CX')
    assert len(calls) == 2
