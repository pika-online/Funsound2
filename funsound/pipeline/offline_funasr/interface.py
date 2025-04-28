
from funsound.engine.funasr.paraformer.interface import SeacoParaformer
from funsound.engine.funasr.punc.interface import CT_Transformer
from funsound.engine.funasr.sv.interface import Embedding
from funsound.brain.translator.interface import Translator
from funsound.pipeline.base import Diarization
from funsound.utils import *
from .core import *



class Noxus(Diarization):
    def __init__(self,
                 taskId, 
                 engine_asr:SeacoParaformer=None,
                 engine_punc:CT_Transformer=None,
                 engine_sv:Embedding=None,
                 translator:Translator=None, 
                 messager = None, 
                 hotwords = [],
                 use_sv=False, 
                 use_trans=False, 
                 source_language="Chinese", 
                 target_language=None):
        super().__init__(taskId, engine_sv, translator,messager, hotwords, use_sv, use_trans, source_language, target_language)
        self.engine_asr = engine_asr
        self.engine_punc = engine_punc
        self.breaker = PuncBreaker(self.engine_punc)

    async def sentence_generator(self, audio_data):
        audio_size = len(audio_data)
        audio_seconds = audio_size/self.sr
        window_seconds = 30
        window_size = int(self.sr*window_seconds)
        window_count = 0

        sentence_count = 0
        sentence_cache = {'tokens':[], 'timestamps':[]}
        for i in range(0, audio_size, window_size):
            # 提取window数据
            s, e = i, min(i + window_size, audio_size)
            window = audio_data[s:e]

            # 语音识别: 热词 + 时间戳 (单窗)
            tmp = await self.engine_asr([window],hotwords=" ".join(self.hotwords))
            result_asr = tmp[1][0]
            for item in result_asr:
                item[1] += window_count*window_seconds
                item[2] += window_count*window_seconds
            result_asr = [item for item in result_asr if item[0] != '<sil>']

            # 标点断句
            sentence_asr = {'tokens':[], 'timestamps':[]}
            for line in result_asr:
                token ,start,end,score = line
                sentence_asr['tokens'].append(token)
                sentence_asr['timestamps'].append([start, end])
            result_punc, sentence_cache = await self.breaker.punc_process(sentence_asr, sentence_cache)

            # 断句分析
            for sentence in result_punc:
                sentenceId = f"{self.taskId}_{sentence_count}"
                text = self.breaker.remove_spaces_between_chinese(" ".join(sentence['tokens']))
                sentence2 = {'start':sentence['timestamps'][0][0],
                           'end':sentence['timestamps'][-1][-1],
                           'text':text,
                           'id':sentenceId}
                sentence_count += 1           
                yield sentence2
                # await asyncio.sleep(0)
            
            window_count += 1

        if sentence_cache['tokens']:
            sentence_cache['tokens'][-1] += '。'
            sentenceId = f"{self.taskId}_{sentence_count}"
            text = self.breaker.remove_spaces_between_chinese(" ".join(sentence_cache['tokens']))
            sentence2 = {'start':sentence_cache['timestamps'][0][0],
                        'end':sentence_cache['timestamps'][-1][-1],
                        'text':text,
                        'id':sentenceId}           
            yield sentence2
            


async def test(engine_paraformer,engine_punc,engine_sv):
    messager = Messager(session_id=generate_random_string(10),debug=True)
    messager.task = '<FUNASR>'
    noxus = Noxus(
        taskId=generate_random_string(10),
        engine_asr=engine_paraformer,
        engine_punc=engine_punc,
        engine_sv=engine_sv,
        translator=Translator(account=llm_account),
        messager=messager,
        hotwords=[],
        use_sv=True,
        use_trans=False,
        source_language=None,
        target_language='Japanese'
    )
    await noxus.run('test.wav')


async def main():
    
    engine_paraformer = SeacoParaformer(
        model_dir=f"{config_paraformer['cache_dir']}/{config_paraformer['model_id']}",
        intra_op_num_threads=config_paraformer['intra_op_num_threads'],
        inter_op_num_threads=config_paraformer['inter_op_num_threads'],
        deviceId=config_paraformer['deviceId']
    )
    engine_punc = CT_Transformer(
        model_dir=f"{config_punc['cache_dir']}/{config_punc['model_id']}",
        intra_op_num_threads=config_punc['intra_op_num_threads'],
        inter_op_num_threads=config_punc['inter_op_num_threads'],
        deviceId=config_punc['deviceId']
    )
    engine_sv = Embedding(
        model_dir=f"{config_sv['cache_dir']}/{config_sv['model_id']}",
        intra_op_num_threads=config_sv['intra_op_num_threads'],
        inter_op_num_threads=config_sv['inter_op_num_threads'],
        deviceId=config_sv['deviceId']
    )


    # 测试多路转写
    n = 3
    tasks = [asyncio.create_task(test(engine_paraformer,engine_punc,engine_sv)) for i in range(n)]
    results = await asyncio.gather(*tasks)
    
    
    


if __name__ == "__main__":

    asyncio.run(main())

