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
    'cache_dir':model_dir,
    "model_id":"QuadraV/funasr_seaco_paraformer_onnx_with_timestamp",
    "intra_op_num_threads":2,
    "inter_op_num_threads":4, # 设置并发数目
    "deviceId": "-1",
}



##################
#    标点
#################
config_punc = {
    'cache_dir':model_dir,
    "model_id":"csukuangfj/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12",
    "intra_op_num_threads":1,
    "inter_op_num_threads":4, # 设置并发数目
    "deviceId": "-1",
}

##################
#    声纹
#################
config_sv = {
    'cache_dir':model_dir,
    "model_id":"QuadraV/speech_eres2net_large_sv_zh-cn_3dspeaker_16k_onnx",
    "intra_op_num_threads":1,
    "inter_op_num_threads":4, # 设置并发数目
    "deviceId": "-1",
}

##################
#    Whisper
#################
config_whisper = {
    'n':2, # 设置并发数目
    "log_dir":'logs/whisper',
    'debug': True,
    'share_session':False,
    'name': 'whisper',
    'cache_dir':model_dir,
    'model_id':"keepitsimple/faster-whisper-large-v3",
    'device':"cuda",
    'compute_type':'float16'
}