from funsound.engine.base import Engine
from funsound.utils import *



class WhisperEngine(Engine):
    def __init__(self, config):
        self.config = config
        super().__init__(
            n=self.config['instances'], 
            log_file=self.config['log_file'], 
            debug=self.config['debug']
        )


    def build_model(self):
        model = faster_whisper.WhisperModel(
                model_size_or_path=f"{self.config['cache_dir']}/{self.config['model_id']}",
                device=self.config['device'],
                compute_type=self.config['compute_type'],
            )
        return model

    def inference_method(self, input_data, model: faster_whisper.WhisperModel, config: dict, message: queue.Queue):
        language = config.get('language', None)
        task = config.get('task', 'transcribe')
        hotwords = config.get('hotwods', [])

        # 调用模型进行识别
        segments, info = model.transcribe(
            input_data,
            condition_on_previous_text=False,
            language=language,
            task=task,
            hotwords= " ".join(hotwords) if hotwords else None
        )

        # 将识别结果依次放入 message 队列
        for i, segment in enumerate(segments):
            asr_result = {
                'start': segment.start,
                'end': segment.end,
                'text': segment.text,
            }
            message.put(('<PROCESS>', asr_result))

    


def test(engine:WhisperEngine):
    """
    测试函数
    """
    input_data = read_audio_file('test1.wav')[:30*16000]
    taskId = generate_random_string(10)

    engine.submit(taskId=taskId, input_data=input_data)

    while True:
        signal, content = engine.messages[taskId].get()
        print(taskId, signal, content)
        if signal == '<END>':
            break
    print("[TEST] Test complete for task:", taskId)


if __name__ == "__main__":

    from funsound.config import *

    # 创建并启动引擎
    engine = WhisperEngine(config=config_whisper)
    engine.start()

    # 并行提交多次测试
    tasks = [threading.Thread(target=test, args=(engine,)) for _ in range(10)]
    for task_thread in tasks:
        task_thread.start()
    for task_thread in tasks:
        task_thread.join()

    # 停止引擎
    engine.stop()
    engine.backend_thread.join()
