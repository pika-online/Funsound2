from funsound.utils import *

class SeacoParaformerPlus(funasr_onnx.SeacoParaformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.SR = 16000
        self.SECONDS_PER_FRAME = 0.02
        self.UPSAMPLE_TIMES = 3

    def greedy_search( self,log_probs, valid_token_len):
        token_ids = log_probs.argmax(axis=-1)
        token_ids_valid = token_ids[:valid_token_len]
        return token_ids_valid

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
            text = self.decode_one(am_scores,valid_token_lens)
            RESULTS.append("".join(text))

            # 时间戳
            timestamp_str, timestamp_raw = funasr_onnx.utils.timestamp_utils.time_stamp_lfr6_onnx(us_peaks, text.copy())
            timestamp_list = self.get_timestamp_list(timestamp_str,am_scores,valid_token_lens)
            TIMESTAMPS.append(timestamp_list)
        return RESULTS,TIMESTAMPS, AM_SCORES, VALID_TOKEN_LENS, US_ALPHAS, US_PEAKS
    
if __name__ == "__main__":

    model = SeacoParaformerPlus(
        model_dir="QuadraV/funasr_seaco_paraformer_onnx_with_timestamp",
        cache_dir="funasr_models",
        quantize=True,
        intra_op_num_threads=4,
        batch_size=1,
        device='-1'
    )
     
    pcm_data = read_audio_file('test1.wav')[:30*16000]
    for _ in range(10):
        with Timer() as t:
            result = model([pcm_data]*5,hotwords='大精树')
        print(result)
        print(t.interval)