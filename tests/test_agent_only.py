import csv
import json

import main as evaluation_main


class _AgentOnlyConfig:
    def get(self, key, default=None):
        values = {
            'agents.http_agent.call_agent_th_max': 1,
            'agents.http_agent.submit_delay': 0,
            'agents.http_agent.call_agent_max_retries': 0,
            'agents.http_agent.call_agent_retry_delay': 0,
            'agents.http_agent.class_config.Diagnosis.is_agent_only': True,
            'agents.http_agent.class_config.PVAssistant.tenant_id_map': {},
            'dataset.placeholder_fill.refresh_entities': False,
        }
        return values.get(key, default)


class _FakeAgent:
    def call_agent(self, question, **kwargs):
        return {'query': question}, 'agent answer', 0.01


def _run_inline(func, items, **kwargs):
    for item in items:
        func(item)


def test_agent_only_stops_after_tmp_is_written(tmp_path, monkeypatch):
    config = _AgentOnlyConfig()
    monkeypatch.setattr(
        evaluation_main.ConfigReader,
        'get_instance',
        lambda: config,
    )
    monkeypatch.setattr(
        evaluation_main,
        'get_enabled_classes',
        lambda selected: ['Diagnosis'],
    )
    monkeypatch.setattr(
        evaluation_main,
        'create_agent',
        lambda class_name: _FakeAgent(),
    )
    monkeypatch.setattr(evaluation_main, 'run_in_thread_pool', _run_inline)
    monkeypatch.setattr(
        evaluation_main,
        'fill_test_cases',
        lambda test_cases, seed: seed,
    )
    monkeypatch.setattr(
        evaluation_main,
        'make_llm_case',
        lambda case: (_ for _ in ()).throw(
            AssertionError('agent-only 不应组装 LLMTestCase')
        ),
    )

    source = tmp_path / 'test_cases_agent_only.csv'
    test_cases = {
        'Diagnosis': [{
            'case_name': str(source),
            'metrics': ['contextual_recall'],
            'csv': [{'query': '故障码 122'}],
        }],
    }
    pipeline = evaluation_main.EvaluationPipeline(
        str(source),
        ['contextual_recall'],
        agent_classes=['Diagnosis'],
    )
    monkeypatch.setattr(pipeline, '_prepare', lambda: test_cases)
    monkeypatch.setattr(
        pipeline,
        '_evaluate',
        lambda cases: (_ for _ in ()).throw(
            AssertionError('agent-only 不应执行评测')
        ),
    )
    monkeypatch.setattr(
        pipeline,
        '_report',
        lambda path, seed: (_ for _ in ()).throw(
            AssertionError('agent-only 不应生成报告')
        ),
    )

    pipeline.run()

    tmp_csv = tmp_path / 'tmp' / 'test_cases_agent_only_tmp.csv'
    tmp_meta = tmp_path / 'tmp' / 'test_cases_agent_only_tmp.meta.json'
    assert tmp_csv.exists()
    assert tmp_meta.exists()
    with tmp_csv.open(encoding='utf-8-sig', newline='') as file:
        rows = list(csv.DictReader(file))
    assert rows[0]['agent_response'] == 'agent answer'
    assert json.loads(tmp_meta.read_text(encoding='utf-8'))['metrics'] == [
        'contextual_recall'
    ]


def test_agent_only_accepts_string_boolean(monkeypatch):
    class _StringConfig:
        def get(self, key, default=None):
            if key.endswith('.is_agent_only'):
                return 'true'
            return default

    monkeypatch.setattr(
        evaluation_main.ConfigReader,
        'get_instance',
        lambda: _StringConfig(),
    )
    pipeline = evaluation_main.EvaluationPipeline(None, None)

    assert pipeline._is_agent_only('Diagnosis') is True
