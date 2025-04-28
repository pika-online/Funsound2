from funsound.utils import *
from funsound.engine.funasr.punc.interface import CT_Transformer
from funsound.pipeline.base import *

class PuncBreaker:
    def __init__(self, engine_punc:CT_Transformer):
        self.pause_threshold = float('inf')
        self.punctuation_list = ["_", "，", "。", "？", "、"]
        self.punctuation_eos_list = ["。", "？","，"]
        self.engine_punc = engine_punc

    def segment_string_to_list( self,input_str):
        """将字符串分割成单词列表，标点符号与前一个单词连接"""
        pattern = re.compile(r'[a-zA-Z\']+|[\u4e00-\u9fff]|[_,，。？、]')
        matches = pattern.findall(input_str)
        result = []
        for i in range(len(matches)):
            if matches[i] in self.punctuation_list and i > 0:
                result[-1] += matches[i]
            else:
                result.append(matches[i])
        return result

    def remove_spaces_between_chinese(self,text):
        # This regex will match spaces between Chinese characters
        pattern = re.compile(r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])')
        while re.search(pattern, text):
            text = re.sub(pattern, r'\1\2', text)
        return text


    async def punc_process(self,sentence_asr,sentence_cache):
        
        tokens, timestamps = sentence_asr['tokens'], sentence_asr['timestamps']
        if not tokens: 
            return [sentence_cache], {'tokens': [], 'timestamps': []}

        # 和前文未完成部分拼接
        if sentence_cache['tokens']:
            tokens = sentence_cache['tokens'] + tokens
            timestamps = sentence_cache['timestamps'] + timestamps
        

        # 英文单词合并
        current_words = []
        current_timestamps = []
        current_text = ''
        current_start = None
        current_end = None
        for token, timestamp in zip(tokens, timestamps):
            start, end = timestamp
            if '@@' in token:
                if current_text == '':
                    current_text = token.replace('@@', '')
                    current_start = start
                    current_end = end
                else:
                    current_text += token.replace('@@', '')
                    current_end = end
            else:
                if current_text != '':
                    if re.findall(r'[a-zA-Z\']',token):  # 检查是否全为英文字符
                        current_text += token
                        current_end = end
                        current_words.append(current_text)
                        current_timestamps.append([current_start, current_end])
                    else:
                        current_words.append(current_text)
                        current_timestamps.append([current_start, current_end])
                        current_words.append(token)
                        current_timestamps.append(timestamp)
                    current_text = ''
                    current_start = None
                    current_end = None
                else:
                    current_words.append(token)
                    current_timestamps.append([start, end])
        
        # 标点预测
        current_text = " ".join(current_words)

        tmp = await self.engine_punc(current_text)
        current_text_with_punc = tmp[0]


        if current_text_with_punc[-1] in self.punctuation_list: # 尾部标点省去
            current_text_with_punc = current_text_with_punc[:-1]
        current_words_with_punc = self.segment_string_to_list(current_text_with_punc)



        # 标点断句
        results = []
        sentence_punc = {'tokens': [], 'timestamps': []}
        for i, (word1, word2, timestamp) in enumerate(zip(current_words, current_words_with_punc, current_timestamps)):
            assert word1 in word2
            sentence_punc['tokens'].append(word2)
            sentence_punc['timestamps'].append(timestamp)
            
            # 停顿过大断句
            if i < (len(current_timestamps) - 1) and (current_timestamps[i + 1][0] - timestamp[1]) > self.pause_threshold:
                results.append(sentence_punc)
                sentence_punc = {'tokens': [], 'timestamps': []}

            # 存在标点断句
            if re.findall(f"[{''.join(self.punctuation_eos_list)}]", word2) and sentence_punc['tokens']:
                results.append(sentence_punc)
                sentence_punc = {'tokens': [], 'timestamps': []}

        # 去除cache的标点
        for i,token in enumerate(sentence_punc['tokens']):
            sentence_punc['tokens'][i] = re.sub(f"[{''.join(self.punctuation_list)}]","",token)
        return results, sentence_punc



