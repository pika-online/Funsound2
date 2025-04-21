from funsound.engine.base import Engine
from funsound.utils import *
from .SeacoParaformer import SeacoParaformerPlus

class ParaformerEngine(Engine):
    def __init__(self, config):
        self.config = config
        super().__init__(
            n=self.config['instances'], 
            log_file=self.config['log_file'], 
            debug=self.config['debug']
        )

    def build_model(self):
        model = SeacoParaformerPlus(
                model_dir=self.config['model_id'],
                cache_dir=self.config['cache_dir'],
                quantize=self.config['quantize'],
                intra_op_num_threads=self.config['intra_op_num_threads'],
                batch_size=1,
                device=self.config['device']
            )
        return model
    
    def inference_method(self, input_data, model:SeacoParaformerPlus, config, message):
        hotwords = config.get('hotwords', [])
        RESULTS,TIMESTAMPS, AM_SCORES, VALID_TOKEN_LENS, US_ALPHAS, US_PEAKS = model([input_data],
                                                                                     hotwords=' '.join(hotwords))
        message.put(('<PROCESS>', TIMESTAMPS[0]))


def test(engine:ParaformerEngine):
    """
    测试函数
    """
    input_data = audio_i2f(read_audio_file('test1.wav'))[:30*16000]
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
    engine = ParaformerEngine(config=config_paraformer)
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