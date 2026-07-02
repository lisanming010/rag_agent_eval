"""日志工厂，基于 logging + QueueHandler/QueueListener 实现线程安全的日志系统

配置从 config.yaml 读取，支持:
- 文件输出（RotatingFileHandler，strftime 文件名模板）
- 控制台输出（StreamHandler）
- QueueHandler + QueueListener 保证多线程下轮转安全
"""

import atexit
import logging
import logging.handlers
import queue
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from tool.config_reader import ConfigReader


class LogFactory:
    """日志工厂（单例），统一管理日志配置与 logger 实例

    用法:
        from tool.log_factory import LogFactory
        logger = LogFactory.get_logger(__name__)
        logger.info("message")
    """

    _instance: Optional["LogFactory"] = None
    _loggers: Dict[str, logging.Logger] = {}
    _listener: Optional[logging.handlers.QueueListener] = None
    _log_queue: Optional[queue.Queue] = None
    _initialized: bool = False

    def __init__(self) -> None:
        if LogFactory._initialized:
            return
        self._conf = ConfigReader.get_instance()
        self._setup()

    def _setup(self) -> None:
        """初始化 QueueHandler + QueueListener 体系"""
        project_root = Path(__file__).parent.parent
        log_dir = self._conf.get("logging.log_dir", "log/")
        log_path = project_root / log_dir
        log_path.mkdir(parents=True, exist_ok=True)

        # 构建 handler 列表，由 QueueListener 单线程消费
        handlers: list[logging.Handler] = []

        file_conf = self._conf.get("logging.file", {})
        if file_conf.get("enabled", True):
            handlers.append(self._build_file_handler(log_path, file_conf))

        console_conf = self._conf.get("logging.console", {})
        if console_conf.get("enabled", True):
            handlers.append(self._build_console_handler(console_conf))

        # 设置 root logger 等级
        root_level = self._conf.get("logging.level", "INFO")
        logging.getLogger().setLevel(_str_to_level(root_level))

        if not handlers:
            LogFactory._initialized = True
            return

        # 创建队列 + QueueListener，将 QueueHandler 挂到 root logger
        queue_maxsize = self._conf.get("logging.queue_maxsize", 0)
        LogFactory._log_queue = queue.Queue(maxsize=queue_maxsize)  # type: ignore[arg-type]

        LogFactory._listener = logging.handlers.QueueListener(
            LogFactory._log_queue,
            *handlers,
            respect_handler_level=True,
        )
        LogFactory._listener.start()

        queue_handler = logging.handlers.QueueHandler(LogFactory._log_queue)
        logging.getLogger().addHandler(queue_handler)

        atexit.register(self._stop_listener)

        LogFactory._initialized = True

    @staticmethod
    def _build_file_handler(log_path: Path, conf: dict) -> logging.Handler:
        """构建 RotatingFileHandler，文件名支持 strftime 模板"""
        filename_template = conf.get("filename", "app_%Y%m%d.log")
        actual_filename = datetime.now().strftime(filename_template)
        file_path = log_path / actual_filename

        max_bytes = conf.get("max_bytes", 10 * 1024 * 1024)
        backup_count = conf.get("backup_count", 5)

        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(_str_to_level(conf.get("level", "DEBUG")))
        file_handler.setFormatter(_build_formatter())
        return file_handler

    @staticmethod
    def _build_console_handler(conf: dict) -> logging.Handler:
        """构建控制台 StreamHandler"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(_str_to_level(conf.get("level", "INFO")))
        console_handler.setFormatter(_build_formatter())
        return console_handler

    @staticmethod
    def _stop_listener() -> None:
        """优雅停止 QueueListener"""
        if LogFactory._listener is not None:
            LogFactory._listener.stop()

    @classmethod
    def get_logger(cls, name: Optional[str] = None) -> logging.Logger:
        """获取指定名称的 logger

        :param name: logger 名称，模块内直接传 ``__name__`` 即可
        :return: logging.Logger 实例
        """
        if cls._instance is None:
            cls._instance = cls()

        if name is None:
            name = "app"

        if name not in cls._loggers:
            cls._loggers[name] = logging.getLogger(name)

        return cls._loggers[name]

    @classmethod
    def reload(cls) -> None:
        """重新加载配置，停止旧 listener 并重建整个日志体系"""
        if cls._listener is not None:
            cls._listener.stop()
            cls._listener = None
        cls._log_queue = None

        root = logging.getLogger()
        for h in root.handlers[:]:
            root.removeHandler(h)

        cls._initialized = False
        cls._instance = None
        cls._loggers.clear()

        ConfigReader.get_instance().reload()
        cls.get_logger()


def _str_to_level(name: str) -> int:
    """字符串 → logging 等级，非法值回退到 DEBUG"""
    level = getattr(logging, name.upper(), None)
    if isinstance(level, int):
        return level
    return logging.DEBUG


def _build_formatter() -> logging.Formatter:
    """从配置读取格式构造 Formatter"""
    conf = ConfigReader.get_instance()
    fmt = conf.get(
        "logging.format",
        "[%(asctime)s] [%(name)s] %(levelname)s - %(message)s",
    )
    datefmt = conf.get("logging.datefmt", "%Y-%m-%d %H:%M:%S")
    return logging.Formatter(fmt=fmt, datefmt=datefmt)
