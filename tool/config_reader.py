"""配置文件读取类"""
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class ConfigReader:
    """用于读取和解析 config.yaml 配置文件的类"""

    _instance: Optional["ConfigReader"] = None

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置读取器

        :params: config_path: 配置文件路径，默认为项目根目录下的 config.yaml
        """
        if config_path is None:
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config.yaml"

        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """加载配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f) or {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项，支持点号分隔的嵌套键

        :params: key: 配置键，如 'judge_llm.Anthropic.model'
        :params: default: 当键不存在时返回的默认值

        :returns: 配置值或默认值
        """
        keys = key.split(".")
        curr_conf = self._config
        for k in keys:
            if isinstance(curr_conf, dict) and k in curr_conf:
                curr_conf = curr_conf[k]
            else:
                return default
        return curr_conf

    def get_all(self) -> Dict[str, Any]:
        """获取完整的配置字典"""
        return self._config

    def reload(self) -> None:
        """重新加载配置文件"""
        self._load_config()

    def __getitem__(self, key: str) -> Any:
        """支持通过 config['key'] 的方式访问配置"""
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value

    def __contains__(self, key: str) -> bool:
        """支持 'key' in config 的判断"""
        return self.get(key) is not None

    @classmethod
    def get_instance(cls, config_path: Optional[str] = None) -> "ConfigReader":
        """获取单例实例""" 
        if cls._instance is None:
            cls._instance = cls(config_path)
        return cls._instance


if __name__ == "__main__":
    config = ConfigReader()
    print("完整配置:", config.get_all())
    print("judge_llm 模型:", config.get("judge_llm.Anthropic.model"))
    print("judge_llm 温度:", config.get("judge_llm.Anthropic.temperature"))
