from funsound.utils import *
from speakerlab.bin.infer_sv import FBank

class Embedding:
    def __init__(self,
                 model_path,
                 intra_op_num_threads=2,
                 inter_op_num_threads=2) -> None:

        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = intra_op_num_threads
        session_options.inter_op_num_threads = inter_op_num_threads
        self.ort_session = onnxruntime.InferenceSession(model_path, session_options)
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
        audio_feat = torch.from_numpy(np.expand_dims(audio_data, axis=0)).float()
        feat = self.feature_extractor(audio_feat).unsqueeze(0).numpy()
        return self.inference(feat)
    
if __name__ == "__main__":
    # 示例用法
    model_path = "funasr_models/iic/speech_eres2net_base_200k_sv_zh-cn_16k-common/model.onnx"
    model = Embedding(model_path)

    audio_data = np.random.randn( 16000 // 2)  # 每条音频持续0.5秒
    print(audio_data.shape)
    embedding = model.gen_embedding(audio_data)
    print(embedding)
