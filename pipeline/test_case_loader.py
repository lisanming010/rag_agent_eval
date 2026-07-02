"""测试用例加载器，负责从 CSV 文件构建测试用例列表"""

import os

from tool.csv_reader import CsvReader
from tool.config_reader import ConfigReader
from tool.log_factory import LogFactory

logger = LogFactory.get_logger(__name__)

def csv_2_case_dict(csv_path: str, metrics: list) -> dict:
    """
    从csv文件创建测试用例

    :param csv_path: csv文件路径
    :param metrics: 指定使用的metrics
    :return: {'csv': [dict], 'case_name': str, 'metrics': [str]}
    """
    test_cases = {}
    test_cases['csv'] = CsvReader(csv_path).read_rows()
    test_cases['case_name'] = str(csv_path)
    test_cases['metrics'] = metrics
    return test_cases


def make_test_case_list(csv_path: str | None, metrics: list | None) -> list[dict]:
    """
    构建测试数据集

    :param csv_path: csv文件的路径，None 则从配置文件读取默认路径
    :param metrics: 传递的metrics，仅当 csv_path 不为空时必传
    :return: [{csv: [dict], case_name: str, metrics: [str]}, ...]
    """
    conf_reader = ConfigReader.get_instance()

    logger.info('读取测试用例集中......')
    test_case_list = []

    # 测试数据集转换
    if csv_path is not None:
        if os.path.exists(csv_path) and os.path.splitext(csv_path)[1].lower() == '.csv':
            if not metrics:
                logger.error('指定测试数据集时-m参数必传')
                raise ValueError('指定测试数据集时-m参数必传')
            test_case_list.append(csv_2_case_dict(csv_path, metrics))
        else:
            logger.error(f"提供的csv文件路径无效或非csv格式文件: {csv_path}")
            raise ValueError(f"提供的csv文件路径无效或非csv格式文件: {csv_path}")
    else:
        default_csv_path = conf_reader.get('dataset.default_dataset_path', None)
        if default_csv_path is None:
            logger.error('执行命令中未指定测试数据集文件，配置文件中也无对应配置，'
                        '请检查配置文件dataset.default_dataset_path配置项')
            raise ValueError(
                '执行命令中未指定测试数据集文件，配置文件中也无对应配置，'
                '请检查配置文件dataset.default_dataset_path配置项'
            )

        if os.path.exists(default_csv_path):
            # 配置文件中指定的是文件夹，默认在当前层级下寻找csv格式文件
            if os.path.isdir(default_csv_path):
                for file in os.listdir(default_csv_path):
                    # 检索test_cases_xxx.csv文件
                    if file.endswith('.csv'):
                        data_set_type = file.removesuffix('.csv').split('_')[-1]
                        if 'test_cases_' not in file:
                            logger.info(f'{file}未被匹配')
                            continue
                        file_metrics = conf_reader.get(
                            f'dataset.dataset_metrics_map.{data_set_type}', None
                        )
                        if file_metrics is None:
                            logger.error(
                                f'{default_csv_path}{file}文件自动匹配metrics方法失败，'
                                f'请检查文件名称是否正确'
                            )
                            raise ValueError(
                                f'{default_csv_path}{file}文件自动匹配metrics方法失败，'
                                f'请检查文件名称是否正确'
                            )
                        file_csv_path = os.path.join(default_csv_path, file)
                        test_case_list.append(csv_2_case_dict(file_csv_path, file_metrics))
                if test_case_list == []:
                    logger.error('未匹配到测试数据集')
                    raise ValueError('未匹配到测试数据集')
            # 配置文件中指定特定的csv文件
            elif os.path.splitext(default_csv_path)[1].lower() == '.csv':
                file_metrics = conf_reader.get('dataset.metrics_if_specify_csv', None)
                if file_metrics is None:
                    logger.error('配置文件中未指定测试数据集使用的metrics')
                    raise ValueError('配置文件中未指定测试数据集使用的metrics')
                test_case_list.append(csv_2_case_dict(default_csv_path, file_metrics))
        else:
            logger.error(f'测试数据集：{default_csv_path}目录不存在')
            raise ValueError(f'测试数据集：{default_csv_path}目录不存在')

    return test_case_list
