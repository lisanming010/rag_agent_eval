import requests
import json
import time
from functools import wraps

class HTTPAgent:
    """
    通过平台接口调用agent
    """
    def __init__(self, base_url, mode: str='FAST'):
        self.base_url = base_url
        self.mode = mode

    def _timing(func):
        # 实际整个方法调用和实际请求调用耗时差距不大，所以使用装饰器计时
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            return (*result, f'{elapsed:.4f}')
        return wrapper
        
    @_timing
    def call_agent(self, question, **kwargs)->list:
        """
        agent调用接口,返回agent的回答

        :param question: 用户输入的问题
        :param kwargs: 其他参数，为后续扩展预留
        :return: agent的回答,json解析后的字典
        """
        data = {
            "query": question,
            "mode": self.mode
        }
        payload = json.dumps(data, ensure_ascii=False)

        # time_start = time.perf_counter()
        response = requests.post(self.base_url, data=payload, headers={'Content-Type': 'application/json'})
        if response.status_code != 200:
            raise RuntimeError(f"Agent调用失败，状态码: {response.status_code}, 响应内容: {response.text}, 实际请求：{response.request.body}")
        # elapsed = time.perf_counter() - time_start
        # 格式化输出
        try:
            response_raw = response.json()
            response_summary = response_raw['data']['summary']
        except json.JSONDecodeError:
            raise RuntimeError(f"响应内容不是有效的JSON格式: {response.text}")
        else:
            return response_raw, response_summary
        
if __name__ == "__main__":
    agent = HTTPAgent(base_url="http://aiforge-serving.tecdo.cn:8584/api/v1/search")
    question = "LP click 这个出价类型会使用哪些类型的预估模型？"
    answer_raw, answer_summary, res_time = agent.call_agent(question)
    print("Agent的回答:", answer_raw)
    print("ageng回答的summary:", answer_summary)
    print('响应时间:', res_time)
    print(type(answer_raw))