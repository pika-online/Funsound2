from funsound.engine.base import *
from funsound.utils import *

class PuncEngine(Engine):
    def __init__(self, config):
        self.config = config
        super().__init__(
            n=self.config['instances'], 
            log_file=self.config['log_file'], 
            debug=self.config['debug']
        )

    def build_model(self):
        model = funasr_onnx.CT_Transformer(
                model_dir=self.config['model_id'],
                cache_dir=self.config['cache_dir'],
                quantize=self.config['quantize'],
                intra_op_num_threads=self.config['intra_op_num_threads'],
                batch_size=1,
                device=self.config['device']
            )
        return model
    
    def inference_method(self, input_data, model:funasr_onnx.CT_Transformer, config, message):
        result = model(input_data)[0]
        message.put(('<PROCESS>', result))




def test(engine, start_time):
    """
    测试函数
    """
    input_data = "hello我的名字叫做李华很高兴认识大家你们"
    taskId = generate_random_string(10)

    engine.submit(taskId=taskId, input_data=input_data)

    while True:
        signal, content = engine.messages[taskId].get()
        if signal == FLAG_PROCESS:
            print(f"[{taskId}] 内容:{content}")
            pass
        if signal in [FLAG_END,FLAG_ERROR]:
            break
    print(f"[{taskId}] 耗时:{time.time() - start_time}")

if __name__ == "__main__":

    from funsound.config import *

    # 创建并启动引擎
    engine = PuncEngine(config=config_punc)
    engine.start()

    # 并行提交多次测试
    start_time = time.time()
    tasks = [threading.Thread(target=test, args=(engine,start_time)) for _ in range(10)]
    for task_thread in tasks:
        task_thread.start()
    for task_thread in tasks:
        task_thread.join()

    # 停止引擎
    engine.stop()