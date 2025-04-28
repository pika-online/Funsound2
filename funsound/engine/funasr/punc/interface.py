from funsound.utils import *
from .utils import *
from ..paraformer.utils import *
from funsound.engine.ort_session import OrtInferSession

class CT_Transformer:


    def __init__(
        self,
        model_dir,
        intra_op_num_threads=4,
        inter_op_num_threads=1,
        deviceId="-1"
    ):


        model_file = os.path.join(model_dir, "model.onnx")
        config_file = os.path.join(model_dir, "config.yaml")
        config = read_yaml(config_file)
        token_list = os.path.join(model_dir, "tokens.json")
        with open(token_list, "r", encoding="utf-8") as f:
            token_list = json.load(f)

        self.converter = TokenIDConverter(token_list)
        self.batch_size = 1
        self.punc_list = config["model_conf"]["punc_list"]
        self.period = 0
        for i in range(len(self.punc_list)):
            if self.punc_list[i] == ",":
                self.punc_list[i] = "，"
            elif self.punc_list[i] == "?":
                self.punc_list[i] = "？"
            elif self.punc_list[i] == "。":
                self.period = i
        self.jieba_usr_dict_path = os.path.join(model_dir, "jieba_usr_dict")
        if os.path.exists(self.jieba_usr_dict_path):
            self.seg_jieba = True
            self.code_mix_split_words_jieba = code_mix_split_words_jieba(self.jieba_usr_dict_path)
        else:
            self.seg_jieba = False

        """
        ONNXRUNTIME SESSION
        """
        self.engine = OrtInferSession(
            model_file=model_file,
            deviceId=deviceId,
            intra_op_num_threads=intra_op_num_threads,
            inter_op_num_threads=inter_op_num_threads,
            )
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=inter_op_num_threads)  # 你可以调大，比如CPU核数
        print("初始化成功")

    async def __call__(self, text: Union[list, str], split_size=20):
        if self.seg_jieba:
            split_text = self.code_mix_split_words_jieba(text)
        else:
            split_text = code_mix_split_words(text)
        split_text_id = self.converter.tokens2ids(split_text)
        mini_sentences = split_to_mini_sentence(split_text, split_size)
        mini_sentences_id = split_to_mini_sentence(split_text_id, split_size)
        assert len(mini_sentences) == len(mini_sentences_id)
        cache_sent = []
        cache_sent_id = []
        new_mini_sentence = ""
        new_mini_sentence_punc = []
        cache_pop_trigger_limit = 200
        for mini_sentence_i in range(len(mini_sentences)):
            mini_sentence = mini_sentences[mini_sentence_i]
            mini_sentence_id = mini_sentences_id[mini_sentence_i]
            mini_sentence = cache_sent + mini_sentence
            mini_sentence_id = np.array(cache_sent_id + mini_sentence_id, dtype="int32")
            data = {
                "text": mini_sentence_id[None, :],
                "text_lengths": np.array([len(mini_sentence_id)], dtype="int32"),
            }
            try:
                outputs = await self.infer(data["text"], data["text_lengths"])
                y = outputs[0]
                punctuations = np.argmax(y, axis=-1)[0]
                assert punctuations.size == len(mini_sentence)
            except :
                logging.warning("error")

            # Search for the last Period/QuestionMark as cache
            if mini_sentence_i < len(mini_sentences) - 1:
                sentenceEnd = -1
                last_comma_index = -1
                for i in range(len(punctuations) - 2, 1, -1):
                    if (
                        self.punc_list[punctuations[i]] == "。"
                        or self.punc_list[punctuations[i]] == "？"
                    ):
                        sentenceEnd = i
                        break
                    if last_comma_index < 0 and self.punc_list[punctuations[i]] == "，":
                        last_comma_index = i

                if (
                    sentenceEnd < 0
                    and len(mini_sentence) > cache_pop_trigger_limit
                    and last_comma_index >= 0
                ):
                    # The sentence it too long, cut off at a comma.
                    sentenceEnd = last_comma_index
                    punctuations[sentenceEnd] = self.period
                cache_sent = mini_sentence[sentenceEnd + 1 :]
                cache_sent_id = mini_sentence_id[sentenceEnd + 1 :].tolist()
                mini_sentence = mini_sentence[0 : sentenceEnd + 1]
                punctuations = punctuations[0 : sentenceEnd + 1]

            new_mini_sentence_punc += [int(x) for x in punctuations]
            words_with_punc = []
            for i in range(len(mini_sentence)):
                if i > 0:
                    if (
                        len(mini_sentence[i][0].encode()) == 1
                        and len(mini_sentence[i - 1][0].encode()) == 1
                    ):
                        mini_sentence[i] = " " + mini_sentence[i]
                words_with_punc.append(mini_sentence[i])
                if self.punc_list[punctuations[i]] != "_":
                    words_with_punc.append(self.punc_list[punctuations[i]])
            new_mini_sentence += "".join(words_with_punc)
            # Add Period for the end of the sentence
            new_mini_sentence_out = new_mini_sentence
            new_mini_sentence_punc_out = new_mini_sentence_punc
            if mini_sentence_i == len(mini_sentences) - 1:
                if new_mini_sentence[-1] == "，" or new_mini_sentence[-1] == "、":
                    new_mini_sentence_out = new_mini_sentence[:-1] + "。"
                    new_mini_sentence_punc_out = new_mini_sentence_punc[:-1] + [self.period]
                elif new_mini_sentence[-1] != "。" and new_mini_sentence[-1] != "？":
                    new_mini_sentence_out = new_mini_sentence + "。"
                    new_mini_sentence_punc_out = new_mini_sentence_punc[:-1] + [self.period]
        return new_mini_sentence_out, new_mini_sentence_punc_out


    def _backend(self, feats: np.ndarray, feats_len: np.ndarray):
        return self.engine([feats, feats_len])


    async def infer(self, feats: np.ndarray, feats_len: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(self.executor, self._backend, feats, feats_len)
        return result


async def test(model,input_data):
    result = await model(input_data)
    print("识别完成")
    return result[0]
    

async def main():

    
    model = CT_Transformer(
        model_dir="models/csukuangfj/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12",
        intra_op_num_threads=1,
        inter_op_num_threads=4,
        deviceId="-1"
    )

    input_data = "hello大家好我的名字叫李华来自广东深圳很高兴来到这里"
    
    with Timer() as t:
        tasks = [asyncio.create_task(test(model,input_data)) for _ in range(10)]
        results = await asyncio.gather(*tasks)
    print(results)
    print('cost:',t.interval)


if __name__ == "__main__":

    asyncio.run(main())