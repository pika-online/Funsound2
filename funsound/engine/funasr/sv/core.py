from funsound.utils import *
import kaldi_native_fbank as knf
import numpy as np

class FBank(object):
    def __init__(self, n_mels, sample_rate, mean_nor: bool = False):
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.mean_nor = mean_nor

        self.opts = knf.FbankOptions()
        self.opts.frame_opts.samp_freq = sample_rate
        self.opts.mel_opts.num_bins = n_mels
        self.opts.frame_opts.dither = 0.0  # 默认不加扰动

    def __call__(self, wav: np.ndarray, dither=0.0) -> np.ndarray:
        assert self.sample_rate == 16000, "只支持16kHz音频"
        assert isinstance(wav, np.ndarray), "输入应为 NumPy 数组"

        # 处理多通道音频：取第一个通道
        if wav.ndim == 2:
            wav = wav[0, :]
        assert wav.ndim == 1, "音频必须是一维或[1, T]"

        # 更新扰动参数
        self.opts.frame_opts.dither = dither

        # 实例化 fbank 提取器
        fbank = knf.OnlineFbank(self.opts)
        fbank.accept_waveform(self.sample_rate, wav.astype(np.float32))
        fbank.input_finished()

        # 提取所有帧
        feats = [fbank.get_frame(i) for i in range(fbank.num_frames_ready)]
        feats = np.stack(feats)  # 形状: [T, N]

        if self.mean_nor:
            feats = feats - np.mean(feats, axis=0, keepdims=True)

        return feats  # 返回 NumPy 数组 [T, n_mels]





class Embedding:
    def __init__(self,
                 model_path,
                 intra_op_num_threads=2,
                 inter_op_num_threads=2) -> None:

        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = intra_op_num_threads
        session_options.inter_op_num_threads = inter_op_num_threads
        # print(onnxruntime.get_available_providers())
        self.ort_session = onnxruntime.InferenceSession(model_path, session_options,providers=['CPUExecutionProvider'])
        self.feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)


    def fixed_audio_size(self,audio_data, min_len=8000, max_len=16000*10):
        audio_length = len(audio_data)
        if audio_length < min_len:
            zeros_to_add = min_len - audio_length
            audio_data = np.pad(audio_data, (0, zeros_to_add), mode='constant')
        elif audio_length > max_len:
            start_index = np.random.randint(0, audio_length - max_len + 1)
            audio_data = audio_data[start_index:start_index + max_len]
        return audio_data

    def inference(self,feats):
        input_name = self.ort_session.get_inputs()[0].name
        outputs = self.ort_session.run(None, {input_name: feats})
        embedding = outputs[0][0]
        return  embedding
    
    def gen_embedding(self,audio_data,min_len=16000, max_len=16000*10):
        audio_data = self.fixed_audio_size(audio_data,min_len,max_len)
        feat = np.expand_dims(self.feature_extractor(audio_data),axis=0).astype('float32')
        return self.inference(feat)
    
if __name__ == "__main__":
    # 示例用法
    model_path = "models/iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k/model.onnx"
    model = Embedding(model_path)

    audio_data = np.random.randn( 16000 // 2)  # 每条音频持续0.5秒
    print(audio_data.shape)
    embedding = model.gen_embedding(audio_data)
    print(embedding)
