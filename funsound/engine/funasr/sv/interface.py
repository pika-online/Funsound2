from .utils import *
from funsound.engine.ort_session import OrtInferSession



class Embedding:
    def __init__(self,
                model_dir,
                intra_op_num_threads=4,
                inter_op_num_threads=1,
                deviceId="-1") -> None:

        self.feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)
        model_file = f"{model_dir}/model.onnx"

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
        

    def fixed_audio_size(self,audio_data, min_len=8000, max_len=16000*10):
        audio_length = len(audio_data)
        if audio_length < min_len:
            zeros_to_add = min_len - audio_length
            audio_data = np.pad(audio_data, (0, zeros_to_add), mode='constant')
        elif audio_length > max_len:
            start_index = np.random.randint(0, audio_length - max_len + 1)
            audio_data = audio_data[start_index:start_index + max_len]
        return audio_data


    
    async def __call__(self,audio_data,min_len=16000, max_len=16000*10):
        audio_data = self.fixed_audio_size(audio_data,min_len,max_len)
        feat = np.expand_dims(self.feature_extractor(audio_data),axis=0).astype('float32')
        return await self.infer(feat)
    
    def _backend(self, feats: np.ndarray):
        return self.engine([feats])


    async def infer(self, feats: np.ndarray):
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(self.executor, self._backend, feats)
        return result


async def test(model,input_data):
    result = await model(input_data)
    print("识别完成")
    return result[0][0]
    

async def main():

    
    model = Embedding(
        model_dir="models/QuadraV/speech_eres2net_large_sv_zh-cn_3dspeaker_16k_onnx",
        intra_op_num_threads=1,
        inter_op_num_threads=4,
        deviceId="-1"
    )

    input_data = np.zeros(16000,dtype='float32')
    
    with Timer() as t:
        tasks = [asyncio.create_task(test(model,input_data)) for _ in range(10)]
        results = await asyncio.gather(*tasks)
    print(results)
    print('cost:',t.interval)


if __name__ == "__main__":

    asyncio.run(main())