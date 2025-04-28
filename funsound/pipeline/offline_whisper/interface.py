
from funsound.engine.whisper.interface import WhisperEngine
from funsound.engine.funasr.sv.interface import Embedding
from funsound.brain.translator.interface import Translator
from funsound.utils import *
from funsound.pipeline.base import Diarization
from funsound.brain.translator.languages import LANGUAGES_WHISPER
from funsound.config import *


class Demacia(Diarization):
    def __init__(self, 
                 taskId, 
                 engine_asr:WhisperEngine, 
                 engine_sv:Embedding=None, 
                 translator:Translator=None, 
                 messager = None, 
                 hotwords = [], 
                 use_sv=False, 
                 use_trans=False, 
                 source_language=None, 
                 target_language=None):
        super().__init__(taskId, engine_sv, translator,messager, hotwords, use_sv, use_trans, source_language, target_language)
        self.engine_asr = engine_asr

    async def G(self,taskId):
        for sentence in self.engine_asr.recv_many(taskId):
            yield sentence

    async def sentence_generator(self, audio_data):
        audio_size = len(audio_data)
        audio_seconds = audio_size/self.sr

        asrId = generate_random_string(10)
        self.engine_asr.submit(taskId=asrId,input_data=audio_data,config={'hotwords':self.hotwords,
                                                                          'task':'transcribe',
                                                                          'language':LANGUAGES_WHISPER[self.source_language] if self.source_language else None})
        sentence_count = 0
        async for sentence in self.G(taskId=asrId):
            sentenceId = f"{self.taskId}_{sentence_count}"
            sentence['id'] = sentenceId
            sentence_count += 1
            yield sentence


async def test(engine_whisper,engine_sv):
    messager = Messager(session_id=generate_random_string(10),debug=True)
    messager.task = '<WHISPER>'
    demacia = Demacia(
        taskId=generate_random_string(10),
        engine_asr=engine_whisper,
        engine_sv=engine_sv,
        translator=Translator(account=llm_account),
        messager=messager,
        hotwords=[],
        use_sv=True,
        use_trans=True,
        source_language=None,
        target_language='English'
    )
    await demacia.run('test.wav')


async def main():
    
    engine_whisper = WhisperEngine(config=config_whisper)
    engine_whisper.start()

    engine_sv = Embedding(
        model_dir=f"{config_sv['cache_dir']}/{config_sv['model_id']}",
        intra_op_num_threads=config_sv['intra_op_num_threads'],
        inter_op_num_threads=config_sv['inter_op_num_threads'],
        deviceId=config_sv['deviceId']
    )
    

    

    # 测试多路转写
    n = 2
    tasks = [asyncio.create_task(test(engine_whisper,engine_sv)) for i in range(n)]
    results = await asyncio.gather(*tasks)
    
    engine_whisper.stop()

if __name__ == "__main__":

    asyncio.run(main())

