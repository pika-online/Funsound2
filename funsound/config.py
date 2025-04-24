model_dir = "models"

##################
#    LLM 账号
#################
llm_account = {
    "api_key": "sk-VwGMQ0RlgLVRk2BN54o7ITHPAkjZyoxFnBR9YvAMz28VJirb",
    "url": "https://api.moonshot.cn/v1/chat/completions",
    "model_id": "moonshot-v1-128k"
}


##################
#    paraformer
#################
config_paraformer = {
    "instances":4, # 实例数目
    "log_file":"logs/engine_paraformer.log",
    "debug": True,
    'cache_dir':model_dir,
    'model_id':"QuadraV/funasr_seaco_paraformer_onnx_with_timestamp",
    'quantize':True,
    'intra_op_num_threads':2,
    'device':-1
}


##################
#    标点
#################
config_punc = {
    'instances':4, # 实例数
    "log_file":'logs/engine_punc.log',
    'debug': False,
    'cache_dir':model_dir,
    'model_id':"csukuangfj/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12",
    'quantize':False,
    'intra_op_num_threads':2,
    'device':-1
}

##################
#    声纹
#################
config_sv = {
    'instances':4, # 实例数
    "log_file":'logs/engine_sv.log',
    'debug': False,
    'cache_dir':model_dir,
    'model_id':"iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k",
    'intra_op_num_threads':2,
    'device':-1
}

##################
#    Whisper
#################
config_whisper = {
    'instances':2, # 实例数
    "log_file":'logs/engine_whisper.log',
    'debug': False,
    'cache_dir':model_dir,
    'model_id':"keepitsimple/faster-whisper-large-v3",
    'device':"cuda",
    'compute_type':'float16'
}