"""测试用例加载器，负责从 CSV 文件构建测试用例列表"""

import json
import os
from pathlib import Path

from agents.factory import get_enabled_classes
from tool.csv_reader import CsvReader
from tool.config_reader import ConfigReader
from tool.log_factory import LogFactory

logger = LogFactory.get_logger(__name__)


def _normalize_csv_fields(rows: list[dict]) -> list[dict]:
    """
    将诊断CSV格式的字段名标准化为管线内部字段名。

    诊断CSV列:  question, kg_path, kg_result
    标准CSV列:  query, expected_behavior, forbidden_behavior

    仅当行中包含 'question' 且不包含 'query' 时触发归一化：
      - question → query
      - kg_path + kg_result → expected_behavior（逗号分隔）

    :param rows: CsvReader.read_rows() 返回的字典列表
    :return: 标准化后的字典列表（原地修改）
    """
    if not rows:
        return rows

    first_row = rows[0]
    if 'question' not in first_row or 'query' in first_row:
        return rows

    for row in rows:
        row['query'] = row.get('question', '')
        row['expected_behavior'] = f"{row.get('kg_path', '')},{row.get('kg_result', '')}"

    logger.info(f"已标准化 {len(rows)} 行诊断CSV字段 (question→query, "
                f"kg_path+kg_result→expected_behavior)")
    return rows


def csv_2_case_dict(csv_path: str, metrics: list) -> dict:
    """
    从csv文件创建测试用例

    :param csv_path: csv文件路径
    :param metrics: 指定使用的metrics
    :return: {'csv': [dict], 'case_name': str, 'metrics': [str]}
    """
    test_cases = {}
    test_cases['csv'] = _normalize_csv_fields(CsvReader(csv_path).read_rows())
    test_cases['case_name'] = str(csv_path)
    test_cases['metrics'] = metrics
    return test_cases


def make_test_case_list(csv_path: str | None, metrics: list | None,
                       agent_classes: list[str] | None = None) -> dict[str, list[dict]]:
    """
    构建测试数据集，按 agent 类名分组返回。

    嗅探模式（csv_path is None 且配置路径为目录）:
      对每个 enabled 的 agent 类，扫描 {test_dir}/{class_name_lower}/ 下
      test_cases_*.csv，通过 dataset_metrics_map 匹配指标。

    CLI 指定模式（csv_path 指向具体 .csv 文件）:
      直接加载该文件，以 '__shared__' 为 key 返回，所有 enabled agent 共享。

    :param csv_path: csv 文件路径，None 则从配置文件读取默认路径
    :param metrics: CLI 指定的 metrics，仅 csv_path 不为空时使用
    :param agent_classes: CLI -a 指定的类名列表，None 时从配置读取 enabled 类
    :return: {class_name: [{csv: [dict], case_name: str, metrics: [str], agent_class: str}, ...], ...}
    """
    conf_reader = ConfigReader.get_instance()

    logger.info('读取测试用例集中......')
    print('读取测试用例集中......')

    # ---- CLI -cp 指定具体文件 ----
    if csv_path is not None:
        if not (os.path.exists(csv_path) and os.path.splitext(csv_path)[1].lower() == '.csv'):
            logger.error(f"提供的csv文件路径无效或非csv格式文件: {csv_path}")
            raise ValueError(f"提供的csv文件路径无效或非csv格式文件: {csv_path}")
        if not metrics:
            logger.error('指定测试数据集时-m参数必传')
            raise ValueError('指定测试数据集时-m参数必传')
        tc = csv_2_case_dict(csv_path, metrics)
        tc['agent_class'] = '__shared__'
        return {'__shared__': [tc]}

    # ---- 自动嗅探模式 ----
    default_csv_path = conf_reader.get('dataset.default_dataset_path', None)
    if default_csv_path is None:
        logger.error('执行命令中未指定测试数据集文件，配置文件中也无对应配置，'
                     '请检查配置文件dataset.default_dataset_path配置项')
        raise ValueError(
            '执行命令中未指定测试数据集文件，配置文件中也无对应配置，'
            '请检查配置文件dataset.default_dataset_path配置项'
        )

    if not os.path.exists(default_csv_path):
        logger.error(f'测试数据集目录不存在: {default_csv_path}')
        raise ValueError(f'测试数据集目录不存在: {default_csv_path}')

    # 配置文件指定了具体 CSV 文件（非目录）
    if not os.path.isdir(default_csv_path):
        if os.path.splitext(default_csv_path)[1].lower() == '.csv':
            file_metrics = conf_reader.get('dataset.metrics_if_specify_csv', None)
            if file_metrics is None:
                logger.error('配置文件中未指定测试数据集使用的metrics')
                raise ValueError('配置文件中未指定测试数据集使用的metrics')
            tc = csv_2_case_dict(default_csv_path, file_metrics)
            tc['agent_class'] = '__shared__'
            return {'__shared__': [tc]}
        else:
            logger.error(f'测试数据集路径无效: {default_csv_path}')
            raise ValueError(f'测试数据集路径无效: {default_csv_path}')

    # 配置文件指定了目录 → 按 enabled 类名扫描子目录
    enabled_classes = get_enabled_classes(agent_classes)
    if not enabled_classes:
        logger.error('没有启用的 agent 类，请检查配置或 -a 参数')
        raise ValueError('没有启用的 agent 类，请检查配置或 -a 参数')

    logger.info(f'启用的 agent 类: {enabled_classes}')
    result: dict[str, list[dict]] = {}

    for class_name in enabled_classes:
        sub_dir = os.path.join(default_csv_path, class_name.lower())
        if not os.path.isdir(sub_dir):
            logger.warning(f'[{class_name}] 子目录不存在: {sub_dir}，跳过')
            continue

        cls_cases: list[dict] = []
        for file in sorted(os.listdir(sub_dir)):
            if not file.startswith('test_cases_') or not file.endswith('.csv'):
                logger.info(f'[{class_name}] {file} 未被匹配（非 test_cases_*.csv 格式）')
                continue

            data_set_type = file.removeprefix('test_cases_').removesuffix('.csv')
            file_metrics = conf_reader.get(
                f'dataset.dataset_metrics_map.{data_set_type}', None
            )
            if file_metrics is None:
                logger.error(
                    f'[{class_name}] {file} 自动匹配metrics失败，'
                    f'请在 dataset.dataset_metrics_map 中配置 "{data_set_type}"'
                )
                raise ValueError(
                    f'[{class_name}] {file} 自动匹配metrics失败，'
                    f'请在 dataset.dataset_metrics_map 中配置 "{data_set_type}"'
                )

            file_path = os.path.join(sub_dir, file)
            tc = csv_2_case_dict(file_path, file_metrics)
            tc['agent_class'] = class_name
            cls_cases.append(tc)
            logger.info(f'[{class_name}] 已匹配: {file} → metrics: {file_metrics}')

        if cls_cases:
            result[class_name] = cls_cases
        else:
            logger.warning(f'[{class_name}] 子目录 {sub_dir} 下未匹配到任何测试数据集')

    if not result:
        logger.error('未匹配到任何测试数据集')
        raise ValueError('未匹配到任何测试数据集')

    return result


def make_tmp_test_case_list(csv_path: str | None, metrics: list | None,
                            agent_classes: list[str] | None = None) -> dict[str, list[dict]]:
    """
    从 tmp 中间文件构建测试用例列表，用于 --resume 重入评测。
    按 agent 类名分组返回，与 make_test_case_list 保持一致的 dict 结构。

    csv_path 支持三种形式:
      - None: 从 config 默认路径下各 enabled 类子目录的 tmp/ 扫描
      - 目录: 扫描 {目录}/tmp/*_tmp.csv → 归入 '__shared__'
      - *_tmp.csv 文件: 直接加载该文件 → 归入 '__shared__'

    :param csv_path: 路径 或 None
    :param metrics: CLI 中 -m 指定的指标，非空时覆盖 meta 文件中的值
    :return: {class_name: [{csv, case_name, metrics, agent_class}, ...], ...}
    """
    conf_reader = ConfigReader.get_instance()

    logger.info('从tmp目录读取测试用例中间文件......')
    print('从tmp目录读取测试用例中间文件......')
    tmp_files: list[tuple[str, dict | None, str]] = []  # (path, meta, agent_class)

    if csv_path is not None:
        if csv_path.endswith('_tmp.csv') and os.path.isfile(csv_path):
            meta = _read_meta_for_tmp(csv_path)
            tmp_files.append((csv_path, meta, '__shared__'))
        elif os.path.isdir(csv_path):
            tmp_dir = Path(csv_path) / 'tmp'
            if not tmp_dir.is_dir():
                raise ValueError(f'tmp目录不存在: {tmp_dir}')
            for f in tmp_dir.glob('*_tmp.csv'):
                meta = _read_meta_for_tmp(str(f))
                tmp_files.append((str(f), meta, '__shared__'))
        else:
            raise ValueError(f'无效路径或文件不存在: {csv_path}')
    else:
        default_path = conf_reader.get('dataset.default_dataset_path', None)
        if default_path is None:
            raise ValueError(
                '未指定测试数据集路径，且配置文件中无dataset.default_dataset_path配置'
            )
        # 遍历各 enabled 类子目录下的 tmp/
        for entry in sorted(os.listdir(default_path)):
            entry_path = os.path.join(default_path, entry)
            if not os.path.isdir(entry_path):
                continue
            # 子目录名映射为类名: diagnosis → Diagnosis
            agent_class = _dir_to_class_name(entry)
            # 若指定了 agent_classes，只加载匹配的类（忽略大小写）
            if agent_classes and agent_class.lower() not in (a.lower() for a in agent_classes):
                continue
            tmp_dir = Path(entry_path) / 'tmp'
            if not tmp_dir.is_dir():
                continue
            for f in tmp_dir.glob('*_tmp.csv'):
                meta = _read_meta_for_tmp(str(f))
                tmp_files.append((str(f), meta, agent_class))

    if not tmp_files:
        raise ValueError('未找到任何 *_tmp.csv 中间文件，请先执行正常评测流程生成中间文件')

    # 按 agent_class 分组
    result: dict[str, list[dict]] = {}
    for tmp_csv, meta, agent_class in tmp_files:
        # metrics 优先级: CLI -m > meta 文件 > config 兜底
        if metrics:
            file_metrics = metrics
        elif meta and meta.get('metrics'):
            file_metrics = meta['metrics']
        else:
            file_metrics = conf_reader.get('dataset.metrics_if_specify_csv', [])

        if not file_metrics:
            logger.warning(f'{tmp_csv} 未找到对应的metrics配置，跳过')
            continue

        # case_name 优先取 meta 中保存的原始路径
        case_name = meta['case_name'] if meta and meta.get('case_name') else tmp_csv

        test_cases = {
            'csv': CsvReader(tmp_csv).read_rows(),
            'case_name': case_name,
            'metrics': file_metrics,
            'agent_class': agent_class,
        }
        result.setdefault(agent_class, []).append(test_cases)
        logger.info(f'[resume] 加载中间文件: {tmp_csv} (class={agent_class}, case_name={case_name}, metrics={file_metrics})')

    return result


def _dir_to_class_name(dir_name: str) -> str:
    """子目录名 → 类名: 'diagnosis' → 'Diagnosis', 'pvassistant' → 'PVAssistant'"""
    # 常见映射: 全小写子目录名 → 首字母大写类名
    return dir_name[0].upper() + dir_name[1:] if dir_name else dir_name


def _read_meta_for_tmp(tmp_csv_path: str) -> dict | None:
    """读取 tmp CSV 对应的 meta.json 文件，不存在时返回 None"""
    meta_path = Path(tmp_csv_path).with_suffix('.meta.json')
    if meta_path.is_file():
        return json.loads(meta_path.read_text(encoding='utf-8'))
    return None
