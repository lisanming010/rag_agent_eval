"""恢复筛选、用例还原和流水线的离线回归。"""

import copy
import json
import sys

import pytest

import main as evaluation_main
from pipeline.resume import prepare_resume
from pipeline import test_case_loader as loader
from tool.csv_writer import CsvWriter
from tool.result_checkpoint import CheckpointError, CheckpointWriter, checkpoint_path, read_committed
from tool.file_utils import mkdir_with_timestamp


def case(number=1, **extra):
    return {'test_id': str(number), 'query': f'q{number}', 'agent_response': f'a{number}',
            'expected_behavior': f'e{number}', 'language': 'en-US', **extra}


def dataset(rows, name='test_cases_demo.csv'):
    return {'case_name': name, 'metrics': ['reverse_validation'], 'csv': rows}


def history(root, rows, agent='Diagnosis', name='result_outputs_demo.csv'):
    path = root / agent.lower() / name
    writer = CheckpointWriter(path, agent)
    writer.commit(rows)
    return path


@pytest.mark.parametrize('success,expected_reused', [(True, 1), ('true', 1), (False, 0), ('', 0), (None, 0)])
def test_only_final_success_controls_reuse(tmp_path, success, expected_reused):
    history(tmp_path, [case(is_success=success, judge_is_success=False, evaluate_error='review warning')])
    prepared, stats = prepare_resume({'Diagnosis': [dataset([case()])]}, str(tmp_path))
    assert stats.reused == expected_reused
    assert stats.reevaluate == 1 - expected_reused
    assert len(prepared['Diagnosis'][0]['csv']) == 1 - expected_reused


def test_partial_history_missing_row_and_uncommitted_tail(tmp_path):
    path = history(tmp_path, [case(1, is_success=True), case(2, is_success=False)])
    CsvWriter(path).append_rows([case(3, is_success=True)])
    before = path.read_bytes(), checkpoint_path(path).read_bytes()
    prepared, stats = prepare_resume({'Diagnosis': [dataset([case(i) for i in range(1, 5)])]}, str(tmp_path))
    assert (stats.total, stats.reused, stats.reevaluate) == (4, 1, 3)
    assert [row['test_id'] for row in prepared['Diagnosis'][0]['csv']] == ['2', '3', '4']
    assert before == (path.read_bytes(), checkpoint_path(path).read_bytes())


def test_full_reevaluation_does_not_read_history(tmp_path, monkeypatch):
    monkeypatch.setattr('pipeline.resume.read_committed', lambda *a: pytest.fail('不应读取历史'))
    prepared, stats = prepare_resume({'Diagnosis': [dataset([case(is_success=True)])]})
    assert stats.reevaluate == 1 and stats.reused == 0
    assert 'is_success' not in prepared['Diagnosis'][0]['csv'][0]


def test_agent_isolation_and_missing_pair(tmp_path):
    # 未选中的旧结果缺少 checkpoint，也不应影响 Diagnosis。
    CsvWriter(tmp_path / 'pvassistant' / 'result_outputs_demo.csv').write_rows([case(is_success=True)])
    _, stats = prepare_resume({'Diagnosis': [dataset([case()])]}, str(tmp_path))
    assert stats.reevaluate == 1 and stats.reused == 0


def test_selected_old_result_fails(tmp_path):
    CsvWriter(tmp_path / 'diagnosis' / 'result_outputs_demo.csv').write_rows([case(is_success=True)])
    with pytest.raises(CheckpointError, match='缺少 checkpoint'):
        prepare_resume({'Diagnosis': [dataset([case()])]}, str(tmp_path))


@pytest.mark.parametrize('kind', ['absent', 'file'])
def test_invalid_root_fails(tmp_path, kind):
    root = tmp_path / kind
    if kind == 'file':
        root.write_text('not a directory', encoding='utf-8')
    with pytest.raises(ValueError, match='--resume-result-dir'):
        prepare_resume({'Diagnosis': [dataset([case()])]}, str(root))


@pytest.mark.parametrize('changed', [{'query': 'different'}, {'agent_response': 'different'},
                                    {'expected_behavior': 'different'}, {'language': 'zh-CN'}])
def test_changed_input_is_not_reused(tmp_path, changed):
    history(tmp_path, [case(is_success=True)])
    _, stats = prepare_resume({'Diagnosis': [dataset([case(**changed)])]}, str(tmp_path))
    assert stats.reused == 0 and stats.reevaluate == 1


def test_query_fallback_and_single_turn_t1_result(tmp_path):
    row = case()
    row.pop('test_id')
    old = evaluation_main._normalize_single_turn_fields({**row, 'is_success': True})
    history(tmp_path, [old])
    _, stats = prepare_resume({'Diagnosis': [dataset([row])]}, str(tmp_path))
    assert stats.all_passed


def multi(missing=False):
    children = [{'用例编号': 'MT1', '_parent_case_id': 'MT1', '_turn': i, '_total_turns': 3,
                 'query': f'm{i}', 'agent_response': '' if missing and i == 2 else f'a{i}',
                 'expected_behavior': f'e{i}', 'language': 'zh-CN', '_source_csv': 'test_cases_demo.csv'}
                for i in range(1, 4)]
    return evaluation_main._merge_multi_turn_rows(children)[0]


def test_multiturn_roundtrip_restores_all_turns_without_mutating_tmp():
    row = multi()
    before = copy.deepcopy(row)
    prepared, stats = prepare_resume({'Diagnosis': [dataset([row])]})
    children = prepared['Diagnosis'][0]['csv']
    assert stats.reevaluate == 1
    assert [child['_turn'] for child in children] == [1, 2, 3]
    assert [child['query'] for child in children] == ['m1', 'm2', 'm3']
    assert [child['agent_response'] for child in children] == ['a1', 'a2', 'a3']
    assert [child['expected_behavior'] for child in children] == ['e1', 'e2', 'e3']
    assert row == before


def test_failed_multiturn_retries_whole_parent(tmp_path):
    row = multi()
    history(tmp_path, [{**row, 'is_success': False, '第1轮对话是否通过': True}])
    prepared, stats = prepare_resume({'Diagnosis': [dataset([row])]}, str(tmp_path))
    assert stats.reevaluate == 1
    assert len(prepared['Diagnosis'][0]['csv']) == 3


def test_incomplete_multiturn_skips_entire_parent():
    prepared, stats = prepare_resume({'Diagnosis': [dataset([multi(missing=True), case()])]})
    tc = prepared['Diagnosis'][0]
    assert stats.unavailable == stats.reevaluate == 1
    assert [row['query'] for row in tc['csv']] == ['q1']
    skipped = tc['skipped_rows'][0]
    assert skipped['is_success'] is False and skipped['_parent_all_pass'] is False
    assert 'evaluate_error(t2)' in skipped
    assert 'evaluate_error(t1)' not in skipped
    assert '第1轮对话是否通过' not in skipped


@pytest.mark.parametrize('response', [None, '', '  \n'])
def test_empty_response_is_unavailable(response):
    prepared, stats = prepare_resume({'Diagnosis': [dataset([case(agent_response=response)])]})
    assert stats.unavailable == 1 and not stats.all_passed
    assert prepared['Diagnosis'][0]['csv'] == []


def pipeline(tmp_path, monkeypatch, result_dir=None):
    class Config:
        def get(self, key, default=None):
            return {'result.save_path': str(tmp_path / 'new'), 'result.write_batch_size': 10,
                    'agents.http_agent.call_agent_th_max': 1}.get(key, default)
    monkeypatch.setattr(evaluation_main.ConfigReader, 'get_instance', lambda: Config())
    instance = evaluation_main.EvaluationPipeline(None, None, resume=True, resume_result_dir=result_dir)
    monkeypatch.setattr(evaluation_main, 'run_in_thread_pool', lambda fn, rows, **kw: [fn(row) for row in rows])
    monkeypatch.setattr(instance, '_report', lambda *a: None)
    return instance


def install_evaluation(monkeypatch, evaluate):
    from evaluator.runner import recompute_overall_success

    async def fake_groups(groups, metrics, on_result, **kwargs):
        for rows in groups:
            kwargs['check_health']()
            batch = {'csv': rows, 'metrics': metrics}
            evaluate(batch)
            recompute_overall_success(batch)
            await on_result(rows)

    monkeypatch.setattr(evaluation_main, 'evaluate_groups', fake_groups)


def forbid(*args, **kwargs):
    pytest.fail('此流程不应被调用')


def test_all_passed_exits_before_assembly_and_output(tmp_path, monkeypatch, capsys):
    old = tmp_path / 'old'
    history(old, [case(is_success=True)])
    instance = pipeline(tmp_path, monkeypatch, str(old))
    monkeypatch.setattr(instance, '_prepare_from_tmp', lambda: {'Diagnosis': [dataset([case()])]})
    monkeypatch.setattr(instance, '_invoke_agents_resume', forbid)
    monkeypatch.setattr(instance, '_evaluate', forbid)
    monkeypatch.setattr(instance, '_report', forbid)
    instance.run()
    assert '全部通过，无需恢复评价' in capsys.readouterr().out
    assert not (tmp_path / 'new').exists()


def test_partial_resume_writes_full_result_and_can_resume_again(tmp_path, monkeypatch):
    old = tmp_path / 'old'
    old_path = history(old, [case(1, is_success=True), case(2, is_success=False)])
    old_before = old_path.read_bytes(), checkpoint_path(old_path).read_bytes()
    instance = pipeline(tmp_path, monkeypatch, str(old))
    inputs = {'Diagnosis': [dataset([case(1), case(2), case(3)])]}
    monkeypatch.setattr(instance, '_prepare_from_tmp', lambda: inputs)
    assembled, evaluated = [], []
    monkeypatch.setattr(evaluation_main, 'make_llm_case', lambda row: assembled.append(row['test_id']))

    def evaluate(batch):
        # 模型调用之前，复用记录已经通过新运行的 checkpoint 提交。
        output = next((tmp_path / 'new').glob('*/diagnosis/*.csv'))
        assert len(read_committed(output, 'Diagnosis').rows) == 1
        evaluated.extend(row['test_id'] for row in batch['csv'])
        for row in batch['csv']:
            row['judge_is_success'] = True

    install_evaluation(monkeypatch, evaluate)
    instance.run()
    assert assembled == evaluated == ['2', '3']
    new_root = next((tmp_path / 'new').iterdir())
    result = new_root / 'diagnosis' / 'result_outputs_demo.csv'
    rows = read_committed(result, 'Diagnosis').rows
    assert [row['test_id'] for row in rows] == ['1', '2', '3']
    assert all(row['is_success'] == 'True' for row in rows)
    assert prepare_resume(inputs, str(new_root))[1].all_passed
    assert old_before == (old_path.read_bytes(), checkpoint_path(old_path).read_bytes())


def test_all_unavailable_still_writes_failure_results(tmp_path, monkeypatch):
    instance = pipeline(tmp_path, monkeypatch)
    monkeypatch.setattr(instance, '_prepare_from_tmp', lambda: {'Diagnosis': [dataset([multi(missing=True)])]})
    for name in ('make_llm_case', 'evaluate_groups'):
        monkeypatch.setattr(evaluation_main, name, forbid)
    instance.run()
    output = next((tmp_path / 'new').glob('*/diagnosis/*.csv'))
    rows = read_committed(output, 'Diagnosis').rows
    assert len(rows) == 1 and rows[0]['is_success'] == 'False'
    assert 'evaluate_error(t2)' in rows[0]


def make_tmp(root, agent='diagnosis', name='demo', meta=True):
    path = root / agent / 'tmp' / f'test_cases_{name}_tmp.csv'
    CsvWriter(path).write_rows([case()])
    if meta:
        path.with_suffix('.meta.json').write_text(json.dumps({
            'case_name': f'Z:/moved/test_cases_{name}.csv', 'metrics': ['reverse_validation'], 'seed': 42,
        }), encoding='utf-8')
    return path


def loader_config(root, monkeypatch):
    class Config:
        def get(self, key, default=None):
            return {'dataset.default_dataset_path': str(root),
                    'dataset.metrics_if_specify_csv': ['reverse_validation']}.get(key, default)
    monkeypatch.setattr(loader.ConfigReader, 'get_instance', lambda: Config())


@pytest.mark.parametrize('explicit', ['none', 'directory', 'file'])
def test_standard_loader_keeps_canonical_agent_and_moved_source(tmp_path, monkeypatch, explicit):
    root = tmp_path / 'test_suite'
    path = make_tmp(root)
    make_tmp(root, 'pvassistant')
    make_tmp(root, 'manual')
    loader_config(root, monkeypatch)
    cp = {'none': None, 'directory': str(path.parent.parent), 'file': str(path)}[explicit]
    loaded = loader.make_tmp_test_case_list(cp, None, ['dIaGnOsIs'])
    assert list(loaded) == ['Diagnosis']
    assert loaded['Diagnosis'][0]['case_name'] == 'Z:/moved/test_cases_demo.csv'
    assert loaded['Diagnosis'][0]['seed'] == 42
    assert loaded['Diagnosis'][0]['tmp_path'] == str(path)


def test_loader_without_a_ignores_enabled_and_unregistered_dirs(tmp_path, monkeypatch):
    root = tmp_path / 'test_suite'
    for agent in ('diagnosis', 'pvassistant', 'manual'):
        make_tmp(root, agent)
    loader_config(root, monkeypatch)
    assert set(loader.make_tmp_test_case_list(None, None)) == {'Diagnosis', 'PVAssistant'}


def test_loader_meta_fallback_and_conflicting_a(tmp_path, monkeypatch):
    root = tmp_path / 'test_suite'
    path = make_tmp(root, meta=False)
    loader_config(root, monkeypatch)
    tc = loader.make_tmp_test_case_list(str(path), None)['Diagnosis'][0]
    assert tc['case_name'].endswith('test_cases_demo.csv')
    with pytest.raises(ValueError, match='不一致'):
        loader.make_tmp_test_case_list(str(path), None, ['PVAssistant'])
    with pytest.raises(ValueError, match='未注册'):
        loader.make_tmp_test_case_list(None, None, ['unknown'])


def test_cli_optional_history_and_invalid_combinations(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['main.py', '--resume', '-a', 'Diagnosis'])
    assert evaluation_main.parse_args().resume_result_dir is None
    monkeypatch.setattr(sys, 'argv', ['main.py', '--resume', '--resume-result-dir', 'result/run'])
    assert evaluation_main.parse_args().resume_result_dir == 'result/run'
    monkeypatch.setattr(sys, 'argv', ['main.py', '--resume-result-dir', 'result/run'])
    with pytest.raises(SystemExit):
        evaluation_main.parse_args()


def test_run_directories_never_collide(tmp_path):
    first = mkdir_with_timestamp(str(tmp_path))
    second = mkdir_with_timestamp(str(tmp_path))
    assert first != second


def test_preflight_validates_every_selected_dataset_before_assembly(tmp_path, monkeypatch):
    old = tmp_path / 'old'
    history(old, [case(is_success=False)])
    CsvWriter(old / 'diagnosis' / 'result_outputs_second.csv').write_rows([case(is_success=True)])
    instance = pipeline(tmp_path, monkeypatch, str(old))
    monkeypatch.setattr(instance, '_prepare_from_tmp', lambda: {'Diagnosis': [
        dataset([case()]), dataset([case()], 'test_cases_second.csv'),
    ]})
    monkeypatch.setattr(instance, '_invoke_agents_resume', forbid)
    monkeypatch.setattr(instance, '_evaluate', forbid)
    with pytest.raises(CheckpointError):
        instance.run()
    assert not (tmp_path / 'new').exists()


def test_full_pipeline_rejudges_previously_passed_input(tmp_path, monkeypatch):
    instance = pipeline(tmp_path, monkeypatch)
    monkeypatch.setattr(instance, '_prepare_from_tmp', lambda: {'Diagnosis': [dataset([case(is_success=True)])]})
    monkeypatch.setattr('pipeline.resume.read_committed', forbid)
    monkeypatch.setattr(evaluation_main, 'create_agent', forbid)
    monkeypatch.setattr(evaluation_main, 'make_llm_case', lambda row: None)
    calls = []

    def evaluate(batch):
        calls.extend(row['query'] for row in batch['csv'])
        for row in batch['csv']:
            row['judge_is_success'] = False

    install_evaluation(monkeypatch, evaluate)
    instance.run()
    assert calls == ['q1']
    output = next((tmp_path / 'new').glob('*/diagnosis/*.csv'))
    assert read_committed(output, 'Diagnosis').rows[0]['is_success'] == 'False'


def test_mixed_schema_remains_consistent_when_tmp_now_contains_only_singles(tmp_path, monkeypatch):
    old = tmp_path / 'old'
    first = evaluation_main._normalize_single_turn_fields(case(1, is_success=True, judge_is_success=True))
    history(old, [first, {**multi(), 'is_success': True}])
    instance = pipeline(tmp_path, monkeypatch, str(old))
    monkeypatch.setattr(instance, '_prepare_from_tmp', lambda: {'Diagnosis': [dataset([case(1), case(2)])]})
    monkeypatch.setattr(evaluation_main, 'make_llm_case', lambda row: None)

    def evaluate(batch):
        for row in batch['csv']:
            row['judge_is_success'] = True

    install_evaluation(monkeypatch, evaluate)
    instance.run()
    output = next((tmp_path / 'new').glob('*/diagnosis/*.csv'))
    rows = read_committed(output, 'Diagnosis').rows
    assert all('query' not in row and row['query(t1)'] for row in rows)
    assert all(row['judge_is_success(t1)'] == 'True' for row in rows)
    from tool.collection_result import CollectionResult
    assert CollectionResult(output).task_success_stats()['judge'] == 100


def test_checkpoint_failure_keeps_only_committed_prefix(tmp_path, monkeypatch):
    import tool.result_checkpoint as module
    original = module.atomic_json

    def fail_second(path, state):
        if state['last_committed_batch'] == 2:
            raise OSError('checkpoint second batch failed')
        return original(path, state)

    monkeypatch.setattr(module, 'atomic_json', fail_second)
    instance = pipeline(tmp_path, monkeypatch)
    monkeypatch.setattr(instance, '_prepare_from_tmp', lambda: {'Diagnosis': [dataset([case(i) for i in range(25)])]})
    monkeypatch.setattr(evaluation_main, 'make_llm_case', lambda row: None)
    sizes = []

    def evaluate(batch):
        sizes.append(len(batch['csv']))
        for row in batch['csv']:
            row['judge_is_success'] = True

    install_evaluation(monkeypatch, evaluate)
    with pytest.raises(RuntimeError, match='checkpoint second batch failed'):
        instance.run()
    assert 20 <= len(sizes) <= 25  # 异步写入失败前允许已有在途评价完成
    output = next((tmp_path / 'new').glob('*/diagnosis/*.csv'))
    snapshot = read_committed(output, 'Diagnosis')
    assert len(snapshot.rows) == 10 and snapshot.has_uncommitted_tail


def test_structured_and_retrieval_inputs_preserve_existing_assembly():
    structured = case(actual_capability_id='cap1', actual_parameters='{}')
    structured.pop('agent_response')
    retrieval = case(retrieved_docs="['doc1', 'doc2']", expected_docs='doc1')
    retrieval.pop('agent_response')
    tc = dataset([retrieval])
    tc['metrics'] = ['mrr']
    prepared, stats = prepare_resume({'DataQA': [dataset([structured])], 'RAGFlowRetriever': [tc]})
    assert stats.reevaluate == 2 and stats.unavailable == 0
    assert prepared['DataQA'][0]['csv'][0]['actual_parameters'] == '{}'
    row = prepared['RAGFlowRetriever'][0]['csv'][0]
    assert row['_need_retrieval'] is True
    assert row['retrieved_docs'] == ['doc1', 'doc2']


def test_missing_retrieval_response_is_unavailable():
    tc = dataset([case()])
    tc['metrics'] = ['reverse_validation', 'mrr']
    prepared, stats = prepare_resume({'PVAssistant': [tc]})
    assert stats.unavailable == 1
    assert prepared['PVAssistant'][0]['csv'] == []


def test_multiturn_global_response_cannot_replace_missing_turn():
    row = multi()
    row.pop('agent_response(t2)')
    row['agent_response'] = 'global answer'
    _, stats = prepare_resume({'Diagnosis': [dataset([row])]})
    assert stats.unavailable == 1


def test_orphan_history_does_not_affect_current_input(tmp_path):
    history(tmp_path, [case(1, is_success=True), {'query': '', 'is_success': False},
                       case(999, is_success=True), case(999, is_success=False)])
    _, stats = prepare_resume({'Diagnosis': [dataset([case(1)])]}, str(tmp_path))
    assert stats.all_passed


def test_duplicate_stable_ids_fail_but_query_duplicates_are_not_checked(tmp_path):
    with pytest.raises(ValueError, match='编号有歧义'):
        prepare_resume({'Diagnosis': [dataset([case(1), case(1, query='different')])]}, str(tmp_path))
    # 无历史模式无需执行关联编号校验，保留全量重评语义。
    assert prepare_resume({'Diagnosis': [dataset([case(1), case(1, query='different')])]})[1].reevaluate == 2
    row = case()
    row.pop('test_id')
    _, stats = prepare_resume({'Diagnosis': [dataset([row, dict(row)])]})
    assert stats.reevaluate == 2


def test_empty_tmp_is_not_misreported_as_all_passed():
    with pytest.raises(ValueError, match='没有可恢复'):
        prepare_resume({'Diagnosis': [dataset([])]})


def test_loader_rejects_tmp_metadata_source_conflict(tmp_path, monkeypatch):
    root = tmp_path / 'test_suite'
    path = make_tmp(root)
    loader_config(root, monkeypatch)
    CsvWriter(path).write_rows([case(_source_csv='test_cases_other.csv')])
    with pytest.raises(ValueError, match='来源不一致'):
        loader.make_tmp_test_case_list(str(path), None)


def test_reports_count_only_the_cases_own_turns(tmp_path):
    from tool.collection_result import CollectionResult
    from tool.get_bad_cases import extract_bad_cases
    path = tmp_path / 'result_outputs_demo.csv'
    CsvWriter(path).write_rows([
        {'_parent_case_id': 'MT1', '_total_turns': 2, 'is_success': True,
         'judge_is_success(t1)': True, 'judge_is_success(t2)': True},
        {'query(t1)': 'single', 'is_success': True, 'judge_is_success(t1)': True},
        {'_parent_case_id': 'MT2', '_total_turns': 3, 'is_success': False,
         'evaluate_error(t2)': 'missing response'},
    ])
    stats = CollectionResult(path).task_success_stats()
    assert stats['total'] == stats['judge'] == pytest.approx(66.67)
    assert extract_bad_cases(str(tmp_path))['demo'] == (1, 3)


def test_unavailable_results_generate_real_report_and_bad_cases(tmp_path, monkeypatch):
    instance = pipeline(tmp_path, monkeypatch)
    monkeypatch.setattr(instance, '_report', evaluation_main.EvaluationPipeline._report.__get__(instance))
    monkeypatch.setattr(instance, '_prepare_from_tmp', lambda: {'Diagnosis': [dataset([multi(missing=True)])]})
    for name in ('make_llm_case', 'evaluate_groups'):
        monkeypatch.setattr(evaluation_main, name, forbid)
    instance.run()
    output = next((tmp_path / 'new').glob('*/diagnosis/result_outputs_*.csv'))
    assert (output.parent / 'test_report.md').is_file()
    assert (output.parent / 'bad_cases_demo.csv').is_file()
    assert not checkpoint_path(output.parent / 'bad_cases_demo.csv').exists()
    assert len(read_committed(output, 'Diagnosis').rows) == 1


@pytest.mark.parametrize('failed_turn', [None, 2])
def test_multiturn_resume_end_to_end(tmp_path, monkeypatch, failed_turn):
    instance = pipeline(tmp_path, monkeypatch)
    monkeypatch.setattr(instance, '_prepare_from_tmp', lambda: {'Diagnosis': [dataset([multi()])]})
    monkeypatch.setattr(evaluation_main, 'make_llm_case', lambda row: None)
    seen = []

    def evaluate(batch):
        seen.extend(row['query'] for row in batch['csv'])
        for row in batch['csv']:
            row['judge_is_success'] = row['_turn'] != failed_turn

    install_evaluation(monkeypatch, evaluate)
    instance.run()
    assert seen == ['m1', 'm2', 'm3']
    output = next((tmp_path / 'new').glob('*/diagnosis/*.csv'))
    rows = read_committed(output, 'Diagnosis').rows
    assert len(rows) == 1
    assert rows[0]['is_success'] == str(failed_turn is None)
    assert rows[0]['第1轮对话是否通过'] == 'True'
    assert rows[0]['第2轮对话是否通过'] == str(failed_turn != 2)


def test_multidataset_and_multiagent_resume_outputs_full_scope(tmp_path, monkeypatch):
    old = tmp_path / 'old'
    history(old, [case(1, is_success=True)], name='result_outputs_en.csv')
    history(old, [case(2, language='zh-CN', is_success=False)], name='result_outputs_zh.csv')
    history(old, [case(3, is_success=True)], agent='PVAssistant')
    inputs = {'Diagnosis': [dataset([case(1)], 'test_cases_en.csv'),
                            dataset([case(2, language='zh-CN')], 'test_cases_zh.csv')],
              'PVAssistant': [dataset([case(3)])]}
    instance = pipeline(tmp_path, monkeypatch, str(old))
    monkeypatch.setattr(instance, '_prepare_from_tmp', lambda: inputs)
    assembled = []
    monkeypatch.setattr(evaluation_main, 'make_llm_case', lambda row: assembled.append(row['test_id']))

    def evaluate(batch):
        for row in batch['csv']:
            row['judge_is_success'] = True

    install_evaluation(monkeypatch, evaluate)
    instance.run()
    assert assembled == ['2']
    new_root = next((tmp_path / 'new').iterdir())
    assert len(list(new_root.glob('*/*.csv'))) == 3
    _, summary = prepare_resume(inputs, str(new_root))
    assert summary.all_passed and summary.reused == 3
