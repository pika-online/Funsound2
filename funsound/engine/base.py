from funsound.utils import *

FLAG_WAIT = '<WAIT>'
FLAG_PROCESS = '<PROCESS>'
FLAG_ERROR = '<ERROR>'
FLAG_END = '<END>'


class Engine:
    def __init__(self, n, log_dir, debug, share_session=False, name=''):
        self.n = n  # 并发数目
        self.log_dir = log_dir  # 日志目录
        self.debug = debug  # 是否显示日志
        self.share_session = share_session  # 是否共享session
        self.name = name  # 引擎名

        mkdir(self.log_dir)

        self.ready = queue.Queue()
        self.request_queue = queue.Queue()
        self.cancel_dict = {}
        self.stop_event = threading.Event()
        self.workers = []
        self.messages = {}  # taskId -> (flag, message_queue)
        self.lock_msg = threading.Lock()

        self.session = self.init_session() if self.share_session else None

    def log(self, log_file, workerId, msg, debug=None):
        """日志输出"""
        debug = self.debug if debug is None else debug
        msg = f"[{get_current_time()}][{self.name}][worker-{workerId}]: {msg}"
        if debug:
            print(msg)
        with open(log_file, 'a', encoding='utf-8') as f:
            print(msg, file=f)

    def init_session(self):
        """初始化模型/会话 (需用户实现)"""
        pass

    def inference(self, session, input_data, config, message):
        """推理逻辑 (需用户实现)"""
        pass

    def work_processor(self, workerId):
        """后台worker"""
        log_file = f"{self.log_dir}/mt-{workerId}.log"
        logger = lambda msg: self.log(log_file, workerId, msg)

        logger("启动监听")
        session = self.session if self.share_session else self.init_session()
        self.ready.put("Ready")

        while not self.stop_event.is_set():
            try:
                item = self.request_queue.get(timeout=1)
            except queue.Empty:
                continue

            if item == "STOP":
                break

            taskId, input_data, config = item
            with self.lock_msg:
                message = self.messages.get(taskId)

            if not message:
                logger(f"任务 {taskId} 不存在，跳过")
                continue

            if self.cancel_dict.pop(taskId, None) is not None:
                logger(f"任务 {taskId} 已被撤销")
                continue

            logger(f"开始推理：{taskId}")
            try:
                with Timer() as t:
                    self.inference(session=session, input_data=input_data, config=config, message=message)
                message.put((FLAG_END, t.interval))
                logger(f"推理成功，耗时：{t.interval:.2f}s")
            except Exception as e:
                message.put((FLAG_ERROR, str(e)))
                logger(f"推理失败：{str(e)}")
                traceback.print_exc()

            logger(f"结束任务：{taskId}")

        logger("停止监听")

    def submit(self, taskId, input_data, config=None):
        """提交推理任务"""
        if config is None:
            config = {}
        self.request_queue.put((taskId, input_data, config))
        with self.lock_msg:
            q = queue.Queue()
            q.put((FLAG_WAIT, 'Waiting for Processor'))
            self.messages[taskId] = q

    def cancel(self, taskId):
        """撤销推理任务"""
        self.cancel_dict[taskId] = True

    def start(self):
        """启动引擎"""
        self.workers = [threading.Thread(target=self.work_processor, args=(i,)) for i in range(self.n)]
        for worker in self.workers:
            worker.start()
        for _ in range(self.n):
            self.ready.get()
        print(f"[{self.name}] 启动完毕，待命中...")

    def stop(self):
        """关闭引擎"""
        for _ in range(self.n):
            self.request_queue.put("STOP")
        self.stop_event.set()
        for worker in self.workers:
            worker.join()

    def recv_one(self, taskId):
        """接收一个完整推理结果"""
        with self.lock_msg:
            message = self.messages.get(taskId)

        if message is None:
            raise ValueError(f"Task {taskId} 不存在")

        ans = []
        while True:
            signal, content = message.get()
            if signal in (FLAG_END, FLAG_ERROR):
                break
            ans = content
        return ans

    def recv_many(self, taskId):
        """接收推理过程中的多个中间结果 (生成器)"""
        with self.lock_msg:
            message = self.messages.get(taskId)

        if message is None:
            raise ValueError(f"Task {taskId} 不存在")

        while True:
            signal, content = message.get()
            if signal in (FLAG_END, FLAG_ERROR):
                break
            if signal == FLAG_PROCESS:
                yield content