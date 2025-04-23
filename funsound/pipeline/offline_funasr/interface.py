
from funsound.engine.funasr.paraformer.interface import ParaformerEngine,Engine
from funsound.engine.funasr.punc.interface import PuncEngine
from funsound.engine.funasr.sv.interface import SVEngine
from funsound.brain.translator.interface import Translator
from funsound.pipeline.base import Diarization,recv_one,recv_many
from funsound.utils import *
from .core import *



class Noxus(Diarization):
    def __init__(self,
                 taskId, 
                 engine_asr:ParaformerEngine=None,
                 engine_punc:PuncEngine=None,
                 engine_sv:SVEngine=None,
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
            asrId = generate_random_string(10)
            self.engine_asr.submit(taskId=asrId,input_data=window,config={'hotwords':self.hotwords})
            result_asr = await recv_one(self.engine_asr,asrId)
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
            




async def main():
    
    engine_paraformer = ParaformerEngine(config=config_paraformer)
    engine_punc = PuncEngine(config=config_punc)
    engine_sv = SVEngine(config=config_sv)
    engine_paraformer.start()
    engine_punc.start()
    engine_sv.start()


    messager = Messager(session_id=generate_random_string(10),debug=True)
    messager.task = '<FUNASR>'
    noxus = Noxus(
        taskId=generate_random_string(10),
        engine_asr=engine_paraformer,
        engine_punc=engine_punc,
        engine_sv=engine_sv,
        messager=messager,
        hotwords=[],
        use_sv=True,
        use_trans=True,
        source_language='Chinese',
        target_language='Japanese'
    )
    await noxus.run('test1.wav')
    


if __name__ == "__main__":

    asyncio.run(main())

