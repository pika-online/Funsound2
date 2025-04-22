import asyncio
import websockets
import json
import os

def clear():
    os.system('clear')
    # os.system('cls') # windows

def string_progress(val):
    n = int(val*20)
    return "#" * n + f"({val*100:.1f}%)"


SUPPORTED_TRANSLATE_LANGUAGES = {
    'funasr':{
        'source_language': ['Chinese'],
        'target_language': ['Chinese','English, Japanese','Korean'],
    },
    'whisper':{
        'source_language': [None,'Chinese','English, Japanese','Korean'],
        'target_language': ['Chinese','English', 'Japanese','Korean'],
    },
}

async def asr(uri, 
              file_path, 
              uid='test_user',
              pipeline='funasr',
              hotwords = [],
              use_sv = False,
              use_trans = False,
              source_language=None, 
              target_language=None,):
    
    async with websockets.connect(uri, max_size=None) as websocket:
        filename = os.path.basename(file_path)

        # Step 1: 发送上传任务和文件名
        message = {
            "uid": uid,
            "task": "upload",
            "data": {
                "filename": filename
            }
        }
        await websocket.send(json.dumps(message))
        response = json.loads(await websocket.recv())
        if response['status']=='error':
            print(f"上传错误:{response['msg']}")
            return


        # Step 2: 开始上传文件
        file_size = os.path.getsize(file_path)
        chunk_size = 1024 * 1024
        uploading = 0
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)  # 每次传输 1MB
                uploading += len(chunk)
                if not chunk:
                    break
                await websocket.send(chunk)
                response = json.loads(await websocket.recv())
                if response['status']=='error':
                    print(f"上传错误:{response['msg']}")
                    return
                clear()
                print(string_progress(uploading/file_size))

        await websocket.send(b'') # 上传完毕
        response = json.loads(await websocket.recv())
        if response['status']=='error':
            print(f"上传错误:{response['msg']}")
            return
        clear()
        print(string_progress(1))



        # Step 3: 开始转写
        if use_trans:
            assert source_language in SUPPORTED_TRANSLATE_LANGUAGES[pipeline]['source_language']
            assert target_language in SUPPORTED_TRANSLATE_LANGUAGES[pipeline]['target_language']

        message = {
            "uid": uid,
            "task": pipeline,
            "data": {
                "source_language": source_language,
                "target_language": target_language,
                "use_sv": use_sv,
                "use_trans": use_trans,
                "hotwords": hotwords
            }
        }
        await websocket.send(json.dumps(message))
        progress = 0.
        display = {}
        while 1:
            response = json.loads(await websocket.recv())
            if response['status']=='error':
                print(f"识别错误:{response['msg']}")
                return
            
            progress = response['progress'] if response['progress'] else progress
            result = response['result'] 
            msg = response['msg'] 
            # 实时asr
            if msg == '<ASR>':
                _id = result['id']
                _start = result['start']
                _end = result['end']
                _text = result['text']
                if _id not in display:
                    display[_id] = {'start':_start, 'end':_end, 'text':_text, 'trans':'', 'role':None}
            elif msg == '<TRANS>':
                for _id in result:
                    display[_id]['trans'] = result[_id]
            elif msg == '<SV>':
                for _id in result:
                    display[_id]['role'] = result[_id]
        

            clear()
            print(string_progress(progress))
            print("-"*100)
            for _id in display:
                print(f"id: {_id} | time:{display[_id]['start']:6.2f} - {display[_id]['end']:6.2f} | role:{display[_id]['role']}")
                print(f"text: {display[_id]['text']}")
                print(f"trans: {display[_id]['trans']}")
                print("-"*50)
            
            if msg == '<END>':
                break



if __name__ == "__main__":

    uri = "ws://js1.blockelite.cn:15576/"  # 根据你的服务端端口修改
    file_path = "test1.wav"    # 替换成你要上传的文件路径

    asyncio.run(asr(uri, 
                    file_path, 
                    pipeline='whisper',
                    use_sv=True,
                    use_trans=True,
                    target_language='English'))
