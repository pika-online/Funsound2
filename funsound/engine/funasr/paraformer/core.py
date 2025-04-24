from funsound.lib import *
from .utils import TokenIDConverter,CharTokenizer,OrtInferSession
from .front_end import WavFrontend

def pad_list(xs, pad_value, max_len=None):
    n_batch = len(xs)
    if max_len is None:
        max_len = max(x.size(0) for x in xs)
    # pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)
    # numpy format
    pad = (np.zeros((n_batch, max_len)) + pad_value).astype(np.int32)
    for i in range(n_batch):
        pad[i, : xs[i].shape[0]] = xs[i]

    return pad

   
# 带时间戳版本
class SeacoParaformer():
    def __init__(self,
                 model_dir,
                 device_id="-1",
                 batch_size = 1,
                 intra_op_num_threads=4) -> None:
        
        model_bb_file = os.path.join(model_dir, "model_quant.onnx")
        model_eb_file = os.path.join(model_dir, "model_eb_quant.onnx")
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
        self.ort_infer_bb = OrtInferSession(
            model_bb_file, device_id, intra_op_num_threads=intra_op_num_threads
        )
        self.ort_infer_eb = OrtInferSession(
            model_eb_file, device_id, intra_op_num_threads=intra_op_num_threads
        )
        self.batch_size = batch_size

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


    def bb_infer(
        self, feats: np.ndarray, feats_len: np.ndarray, bias_embed
    ) -> Tuple[np.ndarray, np.ndarray]:
        outputs = self.ort_infer_bb([feats, feats_len, bias_embed])
        return outputs

    def eb_infer(self, hotwords, hotwords_length):
        outputs = self.ort_infer_eb([hotwords.astype(np.int32), hotwords_length.astype(np.int32)])
        return outputs

    def extract_feat(self, waveform_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
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
    
    def __call__(self, waveform_list: list, hotwords: str = "", **kwargs) -> list:

        # 加载热词编码
        hotwords, hotwords_length = self.proc_hotword(hotwords)
        [bias_embed] = self.eb_infer(hotwords, hotwords_length)
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
            feats, feats_len = self.extract_feat(waveform_list[beg_idx:end_idx])
            # 2.热词编码同步复制
            bias_embed_ = np.repeat(bias_embed, feats.shape[0], axis=0)
            # 3. 解码
            am_scores, valid_token_lens,us_alphas, us_peaks = self.bb_infer(feats, feats_len, bias_embed_)

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

    
def time_stamp_lfr6_onnx(us_cif_peak, char_list, begin_time=0.0, total_offset=-1.5):
    if not len(char_list):
        return []
    START_END_THRESHOLD = 5
    MAX_TOKEN_DURATION = 30
    TIME_RATE = 10.0 * 6 / 1000 / 3  #  3 times upsampled
    cif_peak = us_cif_peak.reshape(-1)
    num_frames = cif_peak.shape[-1]
    if char_list[-1] == '</s>':
        char_list = char_list[:-1]
    # char_list = [i for i in text]
    timestamp_list = []
    new_char_list = []
    # for bicif model trained with large data, cif2 actually fires when a character starts
    # so treat the frames between two peaks as the duration of the former token
    fire_place = np.where(cif_peak>1.0-1e-4)[0] + total_offset  # np format
    num_peak = len(fire_place)
    assert num_peak == len(char_list) + 1 # number of peaks is supposed to be number of tokens + 1
    # begin silence
    if fire_place[0] > START_END_THRESHOLD:
        # char_list.insert(0, '<sil>')
        timestamp_list.append([0.0, fire_place[0]*TIME_RATE])
        new_char_list.append('<sil>')
    # tokens timestamp
    for i in range(len(fire_place)-1):
        new_char_list.append(char_list[i])
        if i == len(fire_place)-2 or MAX_TOKEN_DURATION < 0 or fire_place[i+1] - fire_place[i] < MAX_TOKEN_DURATION:
            timestamp_list.append([fire_place[i]*TIME_RATE, fire_place[i+1]*TIME_RATE])
        else:
            # cut the duration to token and sil of the 0-weight frames last long
            _split = fire_place[i] + MAX_TOKEN_DURATION
            timestamp_list.append([fire_place[i]*TIME_RATE, _split*TIME_RATE])
            timestamp_list.append([_split*TIME_RATE, fire_place[i+1]*TIME_RATE])
            new_char_list.append('<sil>')
    # tail token and end silence
    if num_frames - fire_place[-1] > START_END_THRESHOLD:
        _end = (num_frames + fire_place[-1]) / 2
        timestamp_list[-1][1] = _end*TIME_RATE
        timestamp_list.append([_end*TIME_RATE, num_frames*TIME_RATE])
        new_char_list.append("<sil>")
    else:
        timestamp_list[-1][1] = num_frames*TIME_RATE
    if begin_time:  # add offset time in model with vad
        for i in range(len(timestamp_list)):
            timestamp_list[i][0] = timestamp_list[i][0] + begin_time / 1000.0
            timestamp_list[i][1] = timestamp_list[i][1] + begin_time / 1000.0
    assert len(new_char_list) == len(timestamp_list)
    res_str = ""
    for char, timestamp in zip(new_char_list, timestamp_list):
        res_str += "{} {} {};".format(char, timestamp[0], timestamp[1])
    res = []
    for char, timestamp in zip(new_char_list, timestamp_list):
        if char != '<sil>':
            res.append([int(timestamp[0] * 1000), int(timestamp[1] * 1000)])
    return res_str, res


def process_tokens(tokens):
        def is_chinese_word(word):
            for char in word:
                if not ('\u4e00' <= char <= '\u9fff'):
                    return False
            return True
        result = []
        current_word = ''
        previous_token_type = None  # 'C' for Chinese, 'E' for English

        for token in tokens:
            if is_chinese_word(token):
                # Token is a Chinese character
                if previous_token_type == 'E':
                    result.append(' ')
                result.append(token)
                previous_token_type = 'C'
            else:
                # Token is part of an English word
                token_clean = token.replace('@@', '')
                current_word += token_clean

                if not token.endswith('@@'):
                    # End of the English word
                    if previous_token_type in ('C', 'E') and previous_token_type is not None:
                        result.append(' ')
                    result.append(current_word)
                    current_word = ''
                    previous_token_type = 'E'
        ans = ''.join(result)
        # if ans[-1]==" ":ans = ans[:-1]
        return ans



if __name__ == "__main__":

    
            

    
    """
    离线解码测试
    """
    from funsound.utils import *
    
    pcm_data = audio_i2f(read_audio_file('./test.wav'))[:30*16000]
    
    model = SeacoParaformer(model_dir=r"models/QuadraV/funasr_seaco_paraformer_onnx_with_timestamp")
    for _ in range(10):
        with Timer() as t:
            result = model([pcm_data],hotwords='小鸿小鸿 派9')
        print(t.interval)
        # print(result[1][0])

