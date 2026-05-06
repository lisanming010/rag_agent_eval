import os
import queue
import threading
from pathlib import Path

from tool.csv_writer import CsvWriter


class AsyncResultWriter:
    """
    异步结果写入器,使用独立线程处理结果写入,避免阻塞主测试流程。

    采用生产者-消费者模式:
    - 主线程(生产者):完成评测后立即提交结果到队列
    - 写入线程(消费者):从队列中取出结果并写入文件

    :params: base_path: 结果文件保存的基础路径
    :params: max_queue_size: 队列最大容量,默认 0(无限制)
    """

    def __init__(self, base_path: str, max_queue_size: int = 0):
        self.base_path = Path(base_path)
        self.result_queue = queue.Queue(maxsize=max_queue_size)
        self.writer_thread = None
        self.stop_signal = threading.Event()
        self.error_count = 0
        self.success_count = 0
        self._lock = threading.Lock()

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

                if task is None:  # 毒丸信号,退出
                    break

                test_case, case_name = task

                # 执行写入
                csv_file_name = os.path.basename(case_name)
                result_output = csv_file_name.replace('test_case', 'result_output')
                result_csv_path = os.path.join(self.base_path, result_output)

                csv_writer = CsvWriter(str(result_csv_path))
                csv_writer.write_rows(test_case['csv'])

                with self._lock:
                    self.success_count += 1

                print(f"结果已写入: {result_output}")

                self.result_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                with self._lock:
                    self.error_count += 1
                print(f"写入失败: {e}")
                self.result_queue.task_done()

    def submit(self, test_case: dict, case_name: str):
        """
        提交结果到写入队列

        :params: test_case: 测试用例字典,包含 'csv' 键
        :params: case_name: 用例名称,用于生成输出文件名
        """
        if self.writer_thread is None or not self.writer_thread.is_alive():
            raise RuntimeError("写入线程未启动,请先调用 start()")

        self.result_queue.put((test_case, case_name))

    def wait_and_stop(self):
        """等待所有写入完成并停止线程"""
        print("等待所有结果写入完成...")
        self.result_queue.join()  # 等待队列清空
        self.result_queue.put(None)  # 发送毒丸信号
        self.writer_thread.join()  # 等待线程退出

        print(f"所有结果已写入完成 (成功: {self.success_count}, 失败: {self.error_count})")

    def get_stats(self) -> dict:
        """
        获取写入统计信息

        :retruen: 包含成功数、失败数、队列剩余任务数的字典
        """
        with self._lock:
            return {
                "success": self.success_count,
                "error": self.error_count,
                "pending": self.result_queue.qsize()
            }
