"""通用并发工具"""

from concurrent.futures import ThreadPoolExecutor, as_completed


def run_in_thread_pool(fn, items: list, max_workers: int = 8, task_name: str = "task"):
    """
    通用线程池调度方法，对items列表中的每个元素并发执行fn

    :param fn: 可调用对象，接受单个item作为参数，可通过functools.partial绑定额外参数
    :param items: 待处理的数据列表
    :param max_workers: 最大线程数，默认8
    :param task_name: 任务名称，用于日志输出
    """
    total = len(items)
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fn, item): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            completed += 1
            try:
                future.result()
                print(f"[{completed}/{total}] {task_name}成功")
            except Exception as e:
                print(f"[{completed}/{total}] {task_name}失败: {e}")
