from funsound.utils import *
from .utils import TokenIDConverter,CharTokenizer
from .front_end import WavFrontend
from .utils import *
from funsound.engine.ort_session import OrtInferSession
import concurrent.futures



   
# 并发版本
class SeacoParaformer():
    def __init__(self,
                 model_dir,
                 batch_size = 1,
                 intra_op_num_threads=4,
                 inter_op_num_threads=1,
                 deviceId="-1") -> None:
        
        config_file = os.path.join(model_dir, "config.yaml")
        cmvn_file = os.path.join(model_dir, "am.mvn")
        token_list = os.path.join(model_dir, "tokens.json")
        with open(str(config_file), "rb") as f:
            config = yaml.load(f, Loader=yaml.Loader)
        with open(token_list, "r", encoding="utf-8") as f:
            token_list = json.load(f)
        self.vocab = {}
        for i, token in enumerate(token_list):
            self.vocab[token] = i

        self.converter = TokenIDConverter(token_list)
        self.tokenizer = CharTokenizer()
        self.frontend = WavFrontend(cmvn_file=cmvn_file, **config["frontend_conf"])
        self.batch_size = batch_size

        """
        ONNXRUNTIME SESSION
        """
        self.engine_bb = OrtInferSession(
            model_file=f"{model_dir}/model_quant.onnx",
            deviceId=deviceId,
            intra_op_num_threads=intra_op_num_threads,
            inter_op_num_threads=inter_op_num_threads,
            )
        
        self.engine_eb = OrtInferSession(
                model_file=f"{model_dir}/model_eb_quant.onnx",
                deviceId=deviceId,
                intra_op_num_threads=intra_op_num_threads,
                inter_op_num_threads=inter_op_num_threads,
                )
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=inter_op_num_threads)  # 你可以调大，比如CPU核数
        print("初始化成功")

        

    def proc_hotword(self, hotwords):
        hotwords = hotwords.split(" ")
        hotwords_length = [len(i) - 1 for i in hotwords]
        hotwords_length.append(0)
        hotwords_length = np.array(hotwords_length)

        # hotwords.append('<s>')
        def word_map(word):
            hotwords = []
            for c in word:
                if c not in self.vocab.keys():
                    hotwords.append(8403)
                    logging.warning(
                        "oov character {} found in hotword {}, replaced by <unk>".format(c, word)
                    )
                else:
                    hotwords.append(self.vocab[c])
            return np.array(hotwords)

        hotword_int = [word_map(i) for i in hotwords]
        # import pdb; pdb.set_trace()
        hotword_int.append(np.array([1]))
        hotwords = pad_list(hotword_int, pad_value=0, max_len=10)
        # import pdb; pdb.set_trace()
        return hotwords, hotwords_length

    def _backend_bb(self, feats, feats_len, bias_embed):
        return self.engine_bb([feats, feats_len, bias_embed])

    def _backend_eb(self, hotwords, hotwords_length):
        return self.engine_eb([hotwords.astype(np.int32), hotwords_length.astype(np.int32)])

    async def infer_bb(self, feats: np.ndarray, feats_len: np.ndarray, bias_embed) -> tuple:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(self.executor, self._backend_bb, feats, feats_len, bias_embed)
        return result

    async def infer_eb(self, hotwords: np.ndarray, hotwords_length: np.ndarray):
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(self.executor, self._backend_eb, hotwords, hotwords_length)
        return result


    async def extract_feat(self, waveform_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        feats, feats_len = [], []
        for waveform in waveform_list:
            speech, _ = self.frontend.fbank(waveform)
            feat, feat_len = self.frontend.lfr_cmvn(speech)
            feats.append(feat)
            feats_len.append(feat_len)

        feats = self.pad_feats(feats, np.max(feats_len))
        feats_len = np.array(feats_len).astype(np.int32)
        return feats, feats_len

    @staticmethod
    def pad_feats(feats: List[np.ndarray], max_feat_len: int) -> np.ndarray:
        def pad_feat(feat: np.ndarray, cur_len: int) -> np.ndarray:
            pad_width = ((0, max_feat_len - cur_len), (0, 0))
            return np.pad(feat, pad_width, "constant", constant_values=0)

        feat_res = [pad_feat(feat, feat.shape[0]) for feat in feats]
        feats = np.array(feat_res).astype(np.float32)
        return feats

    def get_timestamp_list(self,timestamp_str,am_scores,valid_token_lens):
        timestamp_list_raw = timestamp_str.split(";")
        timestamp_list = []
        i = 0
        for line in timestamp_list_raw:
            if line:
                token ,start,end = line.split()
                start,end = float(start), float(end)
                score = float('-inf')
                if token != '<sil>':
                    score = am_scores[i][self.converter.token2id[token]]
                    i += 1
                timestamp_list.append([token,start,end, score])
        return timestamp_list
    
    def greedy_search( self,log_probs, valid_token_len):
        token_ids = log_probs.argmax(axis=-1)
        token_ids_valid = token_ids[:valid_token_len]
        return token_ids_valid
    
    """
    线程安全 + 并发复用
    """
    async def __call__(self, waveform_list: list, hotwords: str = "", **kwargs) -> list:

        # 加载热词编码
        hotwords, hotwords_length = self.proc_hotword(hotwords)
        [bias_embed] = await self.infer_eb(hotwords, hotwords_length)
        bias_embed = bias_embed.transpose(1, 0, 2)
        _ind = np.arange(0, len(hotwords)).tolist()
        bias_embed = bias_embed[_ind, hotwords_length.tolist()]
        bias_embed = np.expand_dims(bias_embed, axis=0)

        # onnx推理
        waveform_nums = len(waveform_list)
        RESULTS = []
        TIMESTAMPS = []
        AM_SCORES = []
        VALID_TOKEN_LENS = []
        US_ALPHAS = []
        US_PEAKS = []
        for beg_idx in range(0, waveform_nums, self.batch_size):
            end_idx = min(waveform_nums, beg_idx + self.batch_size)
            # 1.计算mel特征
            feats, feats_len = await self.extract_feat(waveform_list[beg_idx:end_idx])
            # 2.热词编码同步复制
            bias_embed_ = np.repeat(bias_embed, feats.shape[0], axis=0)
            # 3. 解码
            am_scores, valid_token_lens,us_alphas, us_peaks = await self.infer_bb(feats, feats_len, bias_embed_)

            AM_SCORES.extend(am_scores)
            VALID_TOKEN_LENS.extend(valid_token_lens)
            US_ALPHAS.extend(us_alphas)
            US_PEAKS.extend(us_peaks)

        
        for am_scores,valid_token_lens,us_peaks in  zip(AM_SCORES,VALID_TOKEN_LENS,US_PEAKS):
            # 贪心搜索
            token_ids_valid = self.greedy_search(am_scores,valid_token_lens)
            token_chs = self.converter.ids2tokens(token_ids_valid)
            text = process_tokens(token_chs)
            RESULTS.append(text)
            # print(token_chs)
            # print(text)

            # 时间戳
            timestamp_str, timestamp_raw = time_stamp_lfr6_onnx(us_peaks, copy.copy(token_chs))
            timestamp_list = self.get_timestamp_list(timestamp_str,am_scores,valid_token_lens)
            TIMESTAMPS.append(timestamp_list)
        return RESULTS,TIMESTAMPS, AM_SCORES, VALID_TOKEN_LENS, US_ALPHAS, US_PEAKS


async def test(model,pcm_data):
    result = await model([pcm_data],hotwords="阿里巴巴")
    print("识别完成")
    return result[1][0]
    

async def main():

    
    model = SeacoParaformer(
        model_dir="models/QuadraV/funasr_seaco_paraformer_onnx_with_timestamp",
        intra_op_num_threads=2,
        inter_op_num_threads=8,
        deviceId="-1"
    )

    pcm_data = audio_i2f(read_audio_file("test.wav"))[:30*16000]
    
    with Timer() as t:
        tasks = [asyncio.create_task(test(model,pcm_data)) for _ in range(10)]
        results = await asyncio.gather(*tasks)
    print(results)
    print('cost:',t.interval)


if __name__ == "__main__":

    asyncio.run(main())