from .lib import *

class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start

def mkdir(path, reset=False):
    if os.path.exists(path):
        if reset:
            shutil.rmtree(path)
            print(f"Removed existing directory: {path}")
            os.makedirs(path)
    else:
        os.makedirs(path)
        print(f"Directory created: {path}")
    return path

def mkfile(path):
    if not os.path.exists(os.path.dirname(path)):
        mkdir(os.path.dirname(path))
    with open(path,'wt') as f:
        pass
    print(f"File created: {path}")

def get_current_time():
    # 获取当前时间
    current_time = datetime.datetime.now()
    # 格式化时间，精确到秒
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_time

def read_audio_file(audio_file):
    """读取音频文件数据并转换为PCM格式。"""
    ffmpeg_cmd = [
        FFMPEG,
        '-i', audio_file,
        '-f', 's16le',
        '-acodec', 'pcm_s16le',
        '-ar', '16k',
        '-ac', '1',
        'pipe:']
    with subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False) as proc:
        stdout_data, stderr_data = proc.communicate()
    pcm_data = np.frombuffer(stdout_data, dtype=np.int16)
    return pcm_data


def read_audio_bytes(audio_bytes):
    ffmpeg_cmd = [
    FFMPEG,
    '-i', 'pipe:',  
    '-f', 's16le',
    '-acodec', 'pcm_s16le',
    '-ar', '16k',
    '-ac', '1',
    'pipe:' ]
    with subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False) as proc:
        stdout_data, stderr_data = proc.communicate(input=audio_bytes)
    pcm_data = np.frombuffer(stdout_data, dtype=np.int16)
    return pcm_data


def read_audio_with_split(audio_file,sr=16000,window_seconds=30):
    window_size = int(sr*window_seconds)
    audio_data = read_audio_file(audio_file)
    audio_length = len(audio_data)
    windows = []
    for i in range(0, audio_length, window_size):
        s, e = i, min(i + window_size, audio_length)
        window = audio_data[s:e]
        windows.append(window)
    return windows

def generate_random_string(n):
    letters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(letters) for i in range(n))
    return random_string

def audio_f2i(data, width=16):
    """将浮点数音频数据转换为整数音频数据。"""
    data = np.array(data)
    return np.int16(data * (2 ** (width - 1)))

def audio_i2f(data, width=16):
    """将整数音频数据转换为浮点数音频数据。"""
    data = np.array(data)
    return np.float32(data / (2 ** (width - 1)))


class Messager:
    """
    用于封装给客户端发送消息以及日志记录
    """

    def __init__(self, session_id: str, ws:websockets=None,log_file="",debug=False):
        self.session_id = session_id
        self.userId = ''
        self.task = None
        self.ws = ws
        self.log_file = log_file
        self.debug = debug

    def write_log(self, content: str):
        """将日志写入文件"""
        # 为当前 session 创建独立日志文件
        if self.debug:
            print(content)
        if self.log_file:
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"[{now_str}]{content}\n")
                

    async def send(self, status: str, msg: str, progress:float = None, completed:bool=True, result=None):
        """
        给客户端发送 JSON 格式的数据，并将发送的信息记录到日志文件中。
        """
        response = {
            'status': status,
            'session_id':self.session_id,
            'task': self.task,
            'progress':progress,
            'completed':completed,
            'msg': msg,
            'result': result,
        }
        # 记录到日志
        self.write_log(f"Send >>> {response}")
        
        if self.ws:
            try:
                await self.ws.send(json.dumps(response,ensure_ascii=False))
            except:
                traceback.print_exc()
                self.write_log("[发送失败]")

    async def send_bytes(self, data: bytes):
        """
        发送二进制数据
        """
        self.write_log(f"SendBytes >>> {len(data)} bytes")
        if self.ws:
            try:
                await self.ws.send(data)
            except:
                traceback.print_exc()
                self.write_log("[发送失败]")

def extract_json(content):
    # Use regex to find the JSON part within the string
    if "```json" not in content:
        json_string = content
        return json_string
    json_match = re.search(r'```json\s*(?P<content>.*?)```', content, re.DOTALL)
    if json_match:
        json_string = json_match.group("content").strip()  # 提取命名组中的内容并移除多余的空白
        return json_string
    return ""