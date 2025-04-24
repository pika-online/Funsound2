
from funsound.engine.whisper.interface import WhisperEngine
from funsound.engine.funasr.sv.interface import SVEngine
from funsound.brain.translator.interface import Translator
from funsound.utils import *
from funsound.pipeline.base import Diarization,recv_one,recv_many
from funsound.brain.translator.languages import LANGUAGES_WHISPER
from funsound.config import *


class Demacia(Diarization):
    def __init__(self, 
                 taskId, 
                 engine_asr:WhisperEngine, 
                 engine_sv:SVEngine=None, 
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
        for sentence in recv_many(self.engine_asr,taskId):
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

async def main():
    
    engine_whisper = WhisperEngine(config=config_whisper)
    engine_sv = SVEngine(config=config_sv)
    engine_whisper.start()
    engine_sv.start()

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
    


if __name__ == "__main__":

    asyncio.run(main())

