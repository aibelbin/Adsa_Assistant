from langchain_groq import ChatGroq

from RealtimeSTT import AudioToTextRecorder
import asyncio 
from localgroq import sendmessage 
from kokoro import KPipeline
import sounddevice as sd
import localgroq


def main():
    recorder = AudioToTextRecorder()
    pipeline = KPipeline(lang_code='b')


    async def hear():
        while True:
            heard = await recorder.text
            response = await localgroq.sendmessage(heard)
            generator = await pipeline(response, voice='af_heart')
            for i, (gs, ps, audio) in enumerate(generator):
                print(f"Segment {i} -> gs={gs}, ps={ps}")
                sd.play(audio, 24000)  
            
    hear()
if __name__ == "__main__":
    main()



