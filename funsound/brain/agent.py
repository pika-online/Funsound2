from .http import LLM

class Agent:
    def __init__(self,account={}):
        self.llm = LLM(api_key=account['api_key'],
                       model_id=account['model_id'],
                       url=account['url'])