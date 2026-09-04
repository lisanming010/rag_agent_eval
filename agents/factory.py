"""Agent工厂，根据配置创建对应的Agent实例"""

from agents.http_agent import PVAssistant, Diagnosis, DataQA, RAGFlowRetriever
from tool.config_reader import ConfigReader

# 类名 → 类的注册表，扩展新 Agent 类型时在此注册
AGENT_CLASS_MAP = {
    'PVAssistant': PVAssistant,
    'Diagnosis': Diagnosis,
    'DataQA': DataQA,
    'RAGFlowRetriever': RAGFlowRetriever,
}

# 不需要 base_url / business_token 参数的 Agent 类名集合
_SELF_CONFIGURED_AGENTS = {'RAGFlowRetriever'}


def create_agent(class_name: str):
    """
    根据类名创建对应的 Agent 实例

    配置从 agents.http_agent.class_config.<class_name> 读取。

    扩展新 Agent 类型时只需:
      - 实现新的 Agent 类
      - 在 AGENT_CLASS_MAP 中注册
      - 在 config.yaml 的 class_config 中添加对应配置

    对于自配置型 Agent（如 RAGFlowRetriever），无需 endpoint/business_token，
    会直接调用其无参构造，由类内部自行读取配置。

    :param class_name: Agent 类名，如 'PVAssistant'、'Diagnosis'、'RAGFlowRetriever'
    :return: Agent 实例
    """
    conf = ConfigReader.get_instance()
    cls = AGENT_CLASS_MAP.get(class_name)
    if cls is None:
        raise NotImplementedError(f'不支持的 agent 类: {class_name}')

    if class_name in _SELF_CONFIGURED_AGENTS:
        return cls()

    endpoint = conf.get(f'agents.http_agent.class_config.{class_name}.endpoint')
    business_token = conf.get(f'agents.http_agent.class_config.{class_name}.business_token', None)

    return cls(base_url=endpoint, business_token=business_token)


def get_enabled_classes(cli_classes: list[str] | None = None) -> list[str]:
    """
    获取本次应执行的 agent 类列表

    优先级: CLI -a 参数 > 配置文件中 enabled == true 的类

    :param cli_classes: CLI 指定的类名列表，None 时从配置读取
    :return: 应执行的类名列表
    """
    conf = ConfigReader.get_instance()
    if cli_classes:
        return cli_classes

    all_classes = list(conf.get('agents.http_agent.class_config', {}).keys())
    return [
        c for c in all_classes
        if conf.get(f'agents.http_agent.class_config.{c}.enabled', False)
    ]


def is_agent_only_enabled(class_name: str, conf=None) -> bool:
    """判断指定 Agent 是否只执行调用阶段，兼容 YAML bool 与字符串配置。"""
    conf = conf or ConfigReader.get_instance()
    value = conf.get(
        f'agents.http_agent.class_config.{class_name}.is_agent_only',
        False,
    )
    if isinstance(value, str):
        return value.strip().upper() in ('TRUE', 'YES', '1')
    return bool(value)
