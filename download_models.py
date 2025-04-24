from modelscope import snapshot_download
from funsound.utils import * 
MODEL_DIR = "./models"
mkdir(MODEL_DIR)


def install(model_id,key_file):
    key_file = os.path.join(MODEL_DIR,model_id,key_file)
    if not os.path.exists(key_file):
        print(f"下载：{key_file}")
        snapshot_download(model_id,cache_dir=MODEL_DIR)
    else:
        print(f"检查：{key_file}")
    

install("QuadraV/funasr_seaco_paraformer_onnx_with_timestamp","model.onnx")
install("csukuangfj/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12","model.onnx")
install("QuadraV/speech_eres2net_large_sv_zh-cn_3dspeaker_16k_onnx","model.onnx")
install("keepitsimple/faster-whisper-large-v3","model.bin")


    



