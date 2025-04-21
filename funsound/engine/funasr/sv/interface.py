from funsound.engine.base import Engine
from funsound.utils import *
from .Embedding import Embedding

class SVEngine(Engine):
    def __init__(self, config):
        self.config = config
        super().__init__(
            n=self.config['instances'], 
            log_file=self.config['log_file'], 
            debug=self.config['debug']
        )

    def build_model(self):
        model = Embedding(
            model_path=f"{self.config['cache_dir']}/{self.config['model_id']}/model.onnx",
            inter_op_num_threads=2,
            intra_op_num_threads=self.config['intra_op_num_threads']
        )
        return model
    
    def inference_method(self, input_data, model:Embedding, config, message):
        result = model.gen_embedding(input_data)
        message.put(('<PROCESS>', result))


def test(engine:SVEngine):
    """
    测试函数
    """
    input_data = np.random.randn( 16000 // 2)
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
    engine = SVEngine(config_sv)
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