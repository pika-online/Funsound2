from funsound.utils import *

FLAG_WAIT = '<WAIT>'
FLAG_PROCESS = '<PROCESS>'
FLAG_ERROR = '<ERROR>'
FLAG_END = '<END>'



class Engine:
    def __init__(self, n, log_file, debug=False):
        self._pool = queue.Queue()
        self.tasks = queue.Queue()
        self.skip = {}
        self.messages = {}

        self.log_file = log_file
        self.debug = debug
        self.stop_event = False
        self.backend_thread = None
        self.lock = threading.Lock()
        mkfile(self.log_file)

        for i in range(n):
            model_instance = self.init_instance()
            self._pool.put(model_instance)
            self.log(f"[INIT] Instance #{i+1} created and added to pool.")

    def log(self, msg):
        msg = f"[{get_current_time()}]: {msg}"
        if self.debug:
            print(msg)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            print(msg, file=f)

    def init_instance(self):
        """初始化 Model 实例。"""
        try:
            self.log("[MODEL INIT] Initializing Model...")
            model = self.build_model()
            self.log("[MODEL INIT] Model initialization complete.")
            return model
        except Exception as e:
            self.log(f"[MODEL INIT ERROR] Failed to initialize Model: {e}")
            traceback.print_exc()
            raise e

    def _inference(self, input_data, model, config, message):
        """
        执行具体的识别推理流程，并将结果通过 message 队列传回。
        """
        try:
            self.log("[INFERENCE] Starting inference...")
            self.inference_method(input_data, model, config, message)
            self.log("[INFERENCE] Inference finished.")
        except Exception as e:
            self.log(f"[INFERENCE ERROR] An error occurred during inference: {e}")
            traceback.print_exc()
            message.put((FLAG_ERROR, f"{e}"))
        finally:
            # 任务完成后，将模型放回到池中并发送结束消息
            self._pool.put(model)
            message.put((FLAG_END, 'Inference completed'))


    def _backend(self):
        """
        后台线程，负责从 tasks 队列中获取任务并分配给空闲的模型实例进行推理。
        """
        self.log("[ENGINE] Backend thread started.")
        while not self.stop_event:
            try:
                task = self.tasks.get(timeout=0.1)
            except queue.Empty:
                task = None

            if task:
                taskId, input_data, config = task
                if not self.skip.get(taskId, False):
                    model = self._pool.get()
                    _process = threading.Thread(
                            target=self._inference,
                            args=(input_data, model, config, self.messages[taskId])
                        )
                    _process.start()
                else:
                    self.log(f"[ENGINE] Task {taskId} was skipped.")
        self.log("[ENGINE] Backend thread stopped.")

    def submit(self, taskId, input_data, config={}):
        """
        提交一个推理任务。
        """
        if config is None:
            config = {}

        self.skip[taskId] = False
        self.messages[taskId] = queue.Queue()
        self.messages[taskId].put(('<WAIT>', 'Waiting for Processor'))
        self.tasks.put((taskId, input_data, config))
        self.log(f"[SUBMIT] Task {taskId} submitted.")

    def cancel(self, taskId):
        """
        取消已提交但尚未处理的任务。
        """
        self.skip[taskId] = True
        self.log(f"[CANCEL] Task {taskId} has been canceled.")


    def start(self):
        """启动引擎"""
        self.backend_thread = threading.Thread(target=self._backend)
        self.backend_thread.start()

    def stop(self):
        """关闭引擎"""
        self.stop_event = True
        self.backend_thread.join()


    # TODO: 初始化模型
    def build_model(self):
        return None

    # TODO: 实现推理过程
    def inference_method(self, input_data, model, config, message):
        pass


def recv_one(engine:Engine,taskId):
    ans = []
    while 1:
        signal, content = engine.messages[taskId].get()
        if signal in [FLAG_END,FLAG_ERROR]:
            break
        else:
            ans = content
    return ans


def recv_many(engine:Engine,taskId):
    while 1:
        signal, content = engine.messages[taskId].get()
        if signal in [FLAG_END,FLAG_ERROR]:
            break
        if signal == FLAG_PROCESS:
            yield content