"""通用文件系统工具"""

import os
from datetime import datetime


def mkdir_with_timestamp(base_path: str) -> str:
    """
    创建当前时间戳命名的文件夹，父目录不存在的情况下也会直接创建，返回完整路径

    :param base_path: 父目录
    :return: 父目录 + 时间戳子目录的完整路径
    """
    timestamp = datetime.now().strftime('%m%d%H%M%S')
    output_dir = os.path.join(base_path, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir
