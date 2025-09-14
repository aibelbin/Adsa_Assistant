import sounddevice as sd
import numpy as np
from TTS.api import TTS
import re
import websockets
import asyncio


def split_text(text):
    return re.split(r'(?<=[.,!?]) +', text)

uri = 'ws://192.168.220.7:8000/rc'

async def stream_tts(text, tts_model_name="tts_models/en/ljspeech/tacotron2-DDC", speaker=None):
    print("Loading TTS model...")
    tts = TTS(tts_model_name)

    print("Synthesizing speech...")
    sentences = split_text(text)
    audio_chunks = []

    for i, sentence in enumerate(sentences):
        if not sentence.strip():
            continue

        print(f"[{i+1}/{len(sentences)}] Synthesizing: {sentence.strip()}")

        audio = tts.tts(sentence, speaker=speaker)
        audio = np.array(audio).astype(np.float32)

        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))

        audio_chunks.append(audio)

    full_audio = np.concatenate(audio_chunks)

    print("Playing full speech...")
    sd.play(full_audio, samplerate=22050)
    sd.wait()
    print("Speech finished.")

async def listen_for_text():
    async with websockets.connect(uri, open_timeout=10) as receiver:
        print("Connected to WebSocket server.")
        async for message in receiver:
            print(f"Received message: {message}")
            await stream_tts(message)

if __name__ == "__main__":
    asyncio.run(listen_for_text())
