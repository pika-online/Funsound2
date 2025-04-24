from funsound.engine.base import *
from funsound.utils import *
from .core import CT_Transformer

class PuncEngine(Engine):
    def __init__(self, config):
        self.config = config
        super().__init__(
            n=self.config['instances'], 
            log_file=self.config['log_file'], 
            debug=self.config['debug']
        )

    def build_model(self):
        model = CT_Transformer(
                model_dir=f"{self.config['cache_dir']}/{self.config['model_id']}",
                quantize=self.config['quantize'],
                intra_op_num_threads=self.config['intra_op_num_threads'],
                device=self.config['device']
            )
        return model
    
    def inference_method(self, input_data, model:CT_Transformer, config, message):
        result = model(input_data)[0]
        message.put((FLAG_PROCESS, result))




def test(engine, start_time):
    """
    测试函数
    """
    input_data = "hello我的名字叫做李华很高兴认识大家你们"
    taskId = generate_random_string(10)

    engine.submit(taskId=taskId, input_data=input_data)

    result = recv_one(engine,taskId)
    print(result)
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