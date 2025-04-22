from ..account import llm_account
from ..agent import Agent
from funsound.utils import *

class Translator(Agent):
    def __init__(self, account={}):
        super().__init__(account)
        self.system_prompt = """你是一个专业翻译助手，能够根据上下文将语音识别文本准确翻译成目标语言。请严格遵循以下规则：

1. 输入格式：用户提供的是一个JSON对象，键是字符串ID（如"0","1"），值是要翻译的原文。例如：
{"0": "Hello", "1": "How are you?"}

2. 翻译任务：将每个句子从源语言翻译成目标语言，确保语义准确、自然流畅，并考虑上下文保证连贯性。

3. 输出格式：必须返回一个合法的JSON字符串，符合以下要求：
- 使用双引号，键与输入ID严格一致。
- 内容紧凑，无换行和空格。
- 整个JSON字符串由"```json"和"```"包装，例如：
```json
{"0":"译文0","1":"译文1"}
```

4. 禁止添加任何额外内容，仅输出翻译后的JSON字符串。

示例：
输入：
{"0": "I love basketball.", "1": "It's my passion."}
翻译成法语：
```json
{"0":"J'adore le basket-ball.","1":"C'est ma passion."}
```
"""

    def translate(self, source_language, target_language, content):
        if source_language is None:
            source_language = '未知语言'
        messages = [{'role':'system', "content": self.system_prompt}]
        # 将用户输入转换为标准JSON字符串，确保双引号
        user_input = json.dumps(content, ensure_ascii=False)
        # 明确翻译任务和输入内容
        task_instruction = f"请将以下内容从{source_language}翻译成{target_language}，保持专业且自然的表达：\n{user_input}"
        messages.append({'role': 'user', 'content': task_instruction})
        # 强化输出格式要求
        messages.append({'role': 'system', 'content': "记住：输出必须是紧凑的双引号JSON，且被\"```json\"和\"```\"包括，无任何额外内容！"})
        
        response = self.llm.request_nonstream(messages)
        return response.json().get('choices', [{}])[0].get('message', {}).get('content', '')

if __name__ == "__main__":
    translator = Translator(account=llm_account)
    # content = \
    # {
    #     "0": '大家好,',
    #     "1": '我是NBA球员科比布莱恩特。',
    #     "2": '我从小热爱篮球，',
    #     "3": '一天不打浑身难受。'
    # }
    content = \
    {
        "0": "Mbote na bino nyonso,",
        "1": "Nazo benga ngai Kobe Bryant, mosani ya NBA.",
        "2": "Nalingaka mingi basketbol banda nazalaki mwana moke,",
        "3": "Soki naleki mokolo moko na kosala te, nazalaka na pasi mingi."
        }
    json_str = translator.translate(
        source_language=None,
        target_language='Chinese',
        content=content  # 注意这里直接传字典，由方法内部转换
    )
    json_str = extract_json(json_str)
    print(json.loads(json_str))