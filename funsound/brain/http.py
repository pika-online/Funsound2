from funsound.utils import *

class LLM:
    def __init__(self, api_key, model_id, url) -> None:
        """
        初始化 LLM 类，用于与大语言模型 API 交互。

        参数：
        - api_key: str, API 密钥，用于验证请求。
        - model_id: str, 模型 ID，指定要使用的语言模型。
        - url: str, API 的请求地址。
        """
        self.api_key = api_key
        self.model_id = model_id
        self.url = url
        self.headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.api_key}'}

    def request_stream(self, messages: list[dict]):
        """向 LLM API 发送流式请求。"""
        data = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": 8192,
            "stream": True
        }
        response = requests.post(self.url, headers=self.headers, json=data, stream=True)
        return response

    def request_nonstream(self, messages: list[dict]):
        """向 LLM API 发送非流式请求。"""
        data = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": 8192,
            "stream": False  # added to support Ollama, because it defaults to True
        }
        response = requests.post(self.url, headers=self.headers, json=data)
        return response

    def generator(self, response):
        """实时处理 LLM 响应并提取句子。"""
        for chunk in response.iter_lines():
            if chunk:
                chunk_data = chunk.decode('utf-8')
                if chunk_data.startswith("data: "):
                    chunk_json = chunk_data[len("data: "):]
                    if chunk_json == '[DONE]':
                        break  # 结束标志
                    parsed_data = json.loads(chunk_json)
                    if parsed_data.get('choices'):
                        chunk_message = parsed_data['choices'][0]['delta'].get('content', "")
                        yield chunk_message if chunk_message else ""
