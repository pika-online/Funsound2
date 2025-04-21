from funsound.utils import *
from funsound.engine.whisper.interface import WhisperEngine
from funsound.engine.funasr.paraformer.interface import ParaformerEngine
from funsound.engine.funasr.punc.interface import PuncEngine
from funsound.engine.funasr.sv.interface import SVEngine
from funsound.brain.translator.interface import Translator
from funsound.pipeline.offline_funasr.interface import Noxus
from funsound.pipeline.offline_whisper.interface import Demacia
from funsound.config import *
from .database import *
from websockets import serve

"""
ENGINE
"""
engine_whisper = WhisperEngine(config=config_whisper)
engine_paraformer = ParaformerEngine(config=config_paraformer)
engine_punc = PuncEngine(config=config_punc)
engine_sv = SVEngine(config=config_sv)
engine_whisper.start()
engine_paraformer.start()
engine_punc.start()
engine_sv.start()




USERS = {}
CACHE_DIR = "./cache"

# ============== WebSocket 处理逻辑 ==============
async def handle_client(websocket):
    """
    处理客户端连接
    """
    client_ip, client_port = websocket.remote_address
    address = f"{client_ip}_{client_port}"

    session_id = generate_random_string(20)
    messager = Messager(session_id=session_id, ws=websocket, debug=True)
    message_timestamps = []

    userId = ''
    fileName = ""
    fileHandle = None
    uploaded = False

    try:
        async for message in websocket:
            # ========== QPS 防御检测开始 ==========
            now = time.time()
            message_timestamps.append(now)
            # 只保留最近 1 秒内的消息记录
            while message_timestamps and (now - message_timestamps[0] > 1):
                message_timestamps.pop(0)

            # 如果 1 秒内消息数超过 10，则断开连接
            if len(message_timestamps) > 20:
                print(f"[handle_client] QPS exceeded for {session_id}, closing connection.")
                await messager.send('error', 'QPS limit exceeded, connection will be closed.')
                await websocket.close()
                break
            # ========== QPS 防御检测结束 ==========

            try:
                # 如果是文本消息
                if isinstance(message, str):
                    _message = json.loads(message)
                    uid = _message.get('uid', '')
                    task = _message.get('task', '')
                    data = _message.get('data', {})
                    if not userId:
                        userId = uid if uid else address
                        cache_dir = f"{CACHE_DIR}/{userId}"
                        USERS[userId] = User(userId,cache_dir=cache_dir)
                        messager.log_file = f"{USERS[userId].cache_dir}/log.txt"
                        mkdir(messager.log_file)
                    messager.task = task
                    messager.write_log(f"Recv(Text) <<< {message}")

                    ###########################
                    #          上传文件
                    ###########################
                    if task == 'upload':
                        _fileName = data.get('filename','')
                        if _fileName:
                            await messager.send('success', f'请流式上传数据')
                            fileName = f"{USERS[userId].cache_dir}/{_fileName}"
                            fileHandle = open(fileName,'wb')
                        else:
                            await messager.send('error', f'非法文件名{_fileName}')
                    
                    ###########################
                    #          Funasr
                    ###########################
                    elif task == 'funasr':
                        if uploaded:
                            source_language = data.get('source_language',None)
                            target_language = data.get('target_language',None)
                            use_sv = data.get('use_sv', False)
                            use_trans = data.get('use_trans', False)
                            hotwords = data.get('hotwords', False)

                            noxus = Noxus(
                                taskId=generate_random_string(10),
                                engine_asr=engine_paraformer,
                                engine_punc=engine_punc,
                                engine_sv=engine_sv,
                                translator=Translator(llm_account) if llm_account['api_key'] else None,
                                messager=messager,
                                hotwords=hotwords,
                                use_sv=use_sv,
                                use_trans=use_trans,
                                source_language=source_language,
                                target_language=target_language
                            )
                            await noxus.run(fileName)
                        else:
                            await messager.send('error', f'未上传文件')

                    ###########################
                    #          Whisper
                    ###########################
                    elif task == 'whisper':
                        if uploaded:
                            source_language = data.get('source_language',None)
                            target_language = data.get('target_language',None)
                            use_sv = data.get('use_sv', False)
                            use_trans = data.get('use_trans', False)
                            hotwords = data.get('hotwords', False)

                            demacia = Demacia(
                                taskId=generate_random_string(10),
                                engine_asr=engine_whisper,
                                engine_sv=engine_sv,
                                translator=Translator(llm_account) if llm_account['api_key'] else None,
                                messager=messager,
                                hotwords=hotwords,
                                use_sv=use_sv,
                                use_trans=use_trans,
                                source_language=source_language,
                                target_language=target_language
                            )
                            await demacia.run(fileName)
                        else:
                            await messager.send('error', f'未上传文件')
                    else:
                        await messager.send('error', f'未知任务{task}')



                else:
                    messager.write_log(f"Recv(Bytes) <<< {len(message)} bytes")
                    if not fileHandle:
                        await messager.send('error', f'拒绝接受..')
                    else:
                        if len(message):
                            fileHandle.write(message)
                            await messager.send('success', f'成功接收{len(message)} 字节')
                        else:
                            fileHandle.close()
                            fileHandle = None
                            uploaded = True
                            await messager.send('success', f'传输完毕')
                    

            except Exception as e:
                await messager.send('error', f'发生错误: {e}')
                traceback.print_exc()

    except asyncio.CancelledError:
        print(f"[handle_client] Connection cancelled for {session_id}")
    except Exception as e:
        traceback.print_exc()
    finally:
        pass

# ============== 服务器主函数 ==============
async def main():
    """
    启动WebSocket服务器
    """

    mkdir('logs',reset=True)
    port = int(sys.argv[1])
    async with serve(handle_client, "0.0.0.0", port, max_size=None):
        print(f"WebSocket server started on ws://0.0.0.0:{port}")
        # 无限阻塞
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
