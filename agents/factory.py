"""Agent工厂，根据配置创建对应的Agent实例"""

from agents.http_agent import PVAssistant
from tool.config_reader import ConfigReader

def create_agent():
    """
    根据配置文件 agents.type 创建对应的 Agent 实例

    扩展新 Agent 类型时只需在此添加分支:
      - 实现新的 Agent 类
      - 在 config.yaml 中配置对应的参数
      - 在此添加 elif 分支

    :return: Agent 实例
    """
    conf = ConfigReader.get_instance()
    agent_type = conf.get('agents.type')

    if agent_type == 'http':
        return PVAssistant(
            base_url=conf.get('agents.http_agent.endpoint'),
            business_token=conf.get('agents.http_agent.business_token')
        )

    raise NotImplementedError(f'不支持的 agent 类型: {agent_type}')
