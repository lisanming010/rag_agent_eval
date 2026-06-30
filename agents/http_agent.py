import requests
import json
import time
from functools import wraps

class PVAssistant:
    """
    通过平台接口调用agent
    """
    def __init__(self, base_url:str, business_token:str):
        self.base_url = base_url

        self.header = {
            "X-Trace-Id": "0123456789abcdef0123456789abcdef",
            "X-System-Code": "agent-workbench",
            "X-Business-Token": business_token,
            "X-Platform": "web",
            "X-Tenant-Id": "146085512914162117",
            "Content-Type": "application/json"
        }

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
    def call_agent(self, question:str, user:str="zhangsan", session_id:str="test-sse", res_mode:str='blocking', **kwargs)->list:
        """
        agent调用接口,返回agent的回答

        :param question: 用户输入的问题
        :param res_mode: agent响应方式：streaming|blocking
        :param user: 调用方标记
        :param session_id: 会话ID，用于维护上下文
        :param kwargs: 其他参数，为后续扩展预留
        :return: agent的回答,json解析后的字典
        """
        data = {
            "inputs": {},
            "query": question,
            "response_mode": res_mode,
            "user": user,
            "files": []
        }
        payload = json.dumps(data, ensure_ascii=False)

        self.header['X-Session-Id'] = session_id
        # time_start = time.perf_counter()
        response = requests.post(self.base_url, data=payload, headers=self.header)
        if response.status_code != 200:
            raise RuntimeError(f"Agent调用失败，状态码: {response.status_code}, 响应内容: {response.text}, 实际请求：{response.request.body}")
        # elapsed = time.perf_counter() - time_start
        # 格式化输出
        try:
            response_raw = response.json()
            response_summary = response_raw['content']['data']['markdown']
        except json.JSONDecodeError:
            raise RuntimeError(f"响应内容不是有效的JSON格式: {response.text}")
        else:
            return response_raw, response_summary
        
if __name__ == "__main__":
    agent = PVAssistant(base_url="http://192.168.100.189:32245/chat-messages",
                        business_token="11")
    question = "集中式逆变器发生故障时，对电站发电量有何影响？"
    answer_raw, answer_summary, res_time = agent.call_agent(question)
    print("Agent的回答:", answer_raw)
    print("ageng回答的summary:", answer_summary)
    print('响应时间:', res_time)
    print(type(answer_raw))