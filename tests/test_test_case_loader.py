import pytest

import pipeline.test_case_loader as test_case_loader
from pipeline.test_case_loader import _normalize_csv_fields


class _LoaderConfig:
    def __init__(self, values):
        self.values = values

    def get(self, key, default=None):
        return self.values.get(key, default)


@pytest.mark.parametrize(
    ('kg_path', 'kg_result', 'expected_behavior'),
    [
        ('诊断路径', '诊断结果', '诊断路径,诊断结果'),
        ('', '诊断结果', '诊断结果'),
        ('   ', '诊断结果', '诊断结果'),
        ('诊断路径', '', '诊断路径'),
        (None, None, ''),
    ],
)
def test_normalize_csv_fields_joins_only_non_empty_kg_fields(
    kg_path, kg_result, expected_behavior
):
    rows = [{
        'question': '设备发生了什么故障？',
        'language': 'en-US',
        'kg_path': kg_path,
        'kg_result': kg_result,
    }]

    normalized = _normalize_csv_fields(rows)

    assert normalized[0]['query'] == '设备发生了什么故障？'
    assert normalized[0]['language'] == 'en-US'
    assert normalized[0]['expected_behavior'] == expected_behavior


def test_agent_only_csv_does_not_require_metrics(tmp_path, monkeypatch):
    csv_path = tmp_path / 'test_cases_diagnosis.csv'
    csv_path.write_text('query\n故障码122\n', encoding='utf-8')
    conf = _LoaderConfig({
        'agents.http_agent.class_config.Diagnosis.is_agent_only': True,
    })
    monkeypatch.setattr(
        test_case_loader.ConfigReader,
        'get_instance',
        lambda: conf,
    )
    monkeypatch.setattr(
        test_case_loader,
        'get_enabled_classes',
        lambda selected: selected or ['Diagnosis'],
    )

    result = test_case_loader.make_test_case_list(
        str(csv_path),
        None,
        ['Diagnosis'],
    )

    assert result['__shared__'][0]['metrics'] == []


def test_normal_csv_still_requires_metrics(tmp_path, monkeypatch):
    csv_path = tmp_path / 'test_cases_diagnosis.csv'
    csv_path.write_text('query\n故障码122\n', encoding='utf-8')
    conf = _LoaderConfig({
        'agents.http_agent.class_config.Diagnosis.is_agent_only': False,
    })
    monkeypatch.setattr(
        test_case_loader.ConfigReader,
        'get_instance',
        lambda: conf,
    )
    monkeypatch.setattr(
        test_case_loader,
        'get_enabled_classes',
        lambda selected: selected or ['Diagnosis'],
    )

    with pytest.raises(ValueError, match='-m参数必传'):
        test_case_loader.make_test_case_list(
            str(csv_path),
            None,
            ['Diagnosis'],
        )


def test_agent_only_directory_scan_does_not_require_metrics_mapping(
    tmp_path,
    monkeypatch,
):
    diagnosis_dir = tmp_path / 'diagnosis'
    diagnosis_dir.mkdir()
    (diagnosis_dir / 'test_cases_without_mapping.csv').write_text(
        'query\n故障码122\n',
        encoding='utf-8',
    )
    conf = _LoaderConfig({
        'dataset.default_dataset_path': str(tmp_path),
        'dataset.placeholder_fill': {},
        'agents.http_agent.class_config.Diagnosis.is_agent_only': True,
    })
    monkeypatch.setattr(
        test_case_loader.ConfigReader,
        'get_instance',
        lambda: conf,
    )
    monkeypatch.setattr(
        test_case_loader,
        'get_enabled_classes',
        lambda selected: selected or ['Diagnosis'],
    )

    result = test_case_loader.make_test_case_list(
        None,
        None,
        ['Diagnosis'],
    )

    assert result['Diagnosis'][0]['metrics'] == []
