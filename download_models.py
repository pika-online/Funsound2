from modelscope import snapshot_download
from funsound.utils import * 
from funsound.engine.funasr.sv.export_onnx import export
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
install("iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k","eres2net_large_model.ckpt")
install("keepitsimple/faster-whisper-large-v3","model.bin")


if not os.path.exists(os.path.join(MODEL_DIR,"iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k","model.onnx")):
    print("SV model 导出 onnx ..")
    export(MODEL_DIR,"iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k")
    



