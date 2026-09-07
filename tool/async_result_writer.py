import asyncio
import os
import queue
import threading
from pathlib import Path

from tool.result_checkpoint import CheckpointWriter, csv_value


class AsyncResultWriter:
    """
    异步结果写入器,使用独立线程处理结果写入,避免阻塞主测试流程。

    采用生产者-消费者模式:
    - 主线程(生产者):完成评测后立即提交结果到队列
    - 写入线程(消费者):从队列中取出结果并写入文件

    :params: base_path: 结果文件保存的基础路径
    :params: max_queue_size: 等待写入的批次数，默认 2
    """

    def __init__(self, base_path: str, max_queue_size: int = 2):
        self.base_path = Path(base_path)
        self.result_queue = queue.Queue(maxsize=max_queue_size)
        self.writer_thread = None
        self.stop_signal = threading.Event()
        self.error_count = 0
        self.success_count = 0
        self.committed_rows = 0
        self._lock = threading.Lock()
        self._write_error = None
        self._checkpoint_writers = {}

    def start(self):
        """启动异步写入线程"""
        if self.writer_thread is not None and self.writer_thread.is_alive():
            print("警告: 写入线程已在运行")
            return

        self.stop_signal.clear()
        self.writer_thread = threading.Thread(
            target=self._write_worker,
            daemon=True,
            name="AsyncResultWriter"
        )
        self.writer_thread.start()
        print("异步写入线程已启动")

    def _write_worker(self):
        """写入工作线程,从队列中取出结果并写入文件"""
        while not self.stop_signal.is_set() or not self.result_queue.empty():
            try:
                # 从队列获取结果,超时 1 秒避免死锁
                task = self.result_queue.get(timeout=1)

            except queue.Empty:
                continue

            try:
                if task is None:  # 毒丸信号,退出
                    break

                rows, case_name, append = task
                self._raise_if_failed()
                csv_file_name = os.path.basename(case_name)
                result_output = csv_file_name.replace('test_case', 'result_output')
                result_csv_path = os.path.join(self.base_path, result_output)
                if result_csv_path not in self._checkpoint_writers:
                    self._checkpoint_writers[result_csv_path] = CheckpointWriter(
                        result_csv_path, self.base_path.name,
                    )
                elif not append:
                    raise ValueError(f'拒绝覆盖已提交的结果: {result_csv_path}')
                self._checkpoint_writers[result_csv_path].commit(rows)

                with self._lock:
                    self.success_count += 1
                    self.committed_rows += len(rows)
                print(f"结果与 checkpoint 已提交: {result_output}（本批 {len(rows)} 条）", flush=True)
            except Exception as e:
                with self._lock:
                    self.error_count += 1
                    if self._write_error is None:
                        self._write_error = e
                print(f"写入失败: {e}")
            finally:
                self.result_queue.task_done()

    def submit(self, test_case: dict, case_name: str, *, append: bool = False):
        """
        提交结果到写入队列

        :params: test_case: 测试用例字典,包含 'csv' 键
        :params: case_name: 用例名称,用于生成输出文件名
        :params: append: 是否追加本运行已提交的结果，False 仅用于首次创建
        """
        if self.writer_thread is None or not self.writer_thread.is_alive():
            raise RuntimeError("写入线程未启动,请先调用 start()")

        self._raise_if_failed()
        rows = self._snapshot(test_case)
        while True:
            self.check_health()
            try:
                self.result_queue.put((rows, case_name, append), timeout=0.05)
                return
            except queue.Full:
                pass

    @staticmethod
    def _snapshot(test_case):
        # 在提交端固定嵌套值的最终 CSV 字符串，后续修改不能改变待写入结果。
        return [{key: csv_value(value) for key, value in row.items()}
                for row in test_case['csv']]

    def check_health(self):
        self._raise_if_failed()
        if self.writer_thread is None or not self.writer_thread.is_alive():
            raise RuntimeError('写入线程未运行')

    async def submit_async(self, test_case: dict, case_name: str, *, append=False):
        """队列满时异步背压，不阻塞事件循环；成功入队与返回之间不 await。"""
        rows = self._snapshot(test_case)
        while True:
            self.check_health()
            try:
                self.result_queue.put_nowait((rows, case_name, append))
                return
            except queue.Full:
                await asyncio.sleep(0.05)

    def _raise_if_failed(self):
        with self._lock:
            error = self._write_error
        if error is not None:
            raise RuntimeError(f"评价结果写入失败: {error}") from error

    def flush(self):
        """等待 CSV 与 checkpoint 均提交；失败交回主线程，不继续评价。"""
        self.result_queue.join()
        self._raise_if_failed()

    def wait_and_stop(self):
        """等待所有写入完成并停止线程"""
        if self.writer_thread is None or not self.writer_thread.is_alive():
            self._raise_if_failed()
            return
        print("等待所有结果写入完成...")
        self.result_queue.join()  # 等待队列清空
        self.result_queue.put(None)  # 发送毒丸信号
        self.writer_thread.join()  # 等待线程退出

        self._raise_if_failed()
        print(f"所有结果已写入完成 (成功: {self.success_count}, 失败: {self.error_count})")

    def get_stats(self) -> dict:
        """
        获取写入统计信息

        :retruen: 包含成功数、失败数、队列剩余任务数的字典
        """
        with self._lock:
            return {
                "success": self.success_count,
                "committed_rows": self.committed_rows,
                "error": self.error_count,
                "pending": self.result_queue.qsize()
            }
