import sounddevice as sd
import numpy as np
from TTS.api import TTS
import re


def split_text(text):
    return re.split(r'(?<=[.,!?]) +', text)


def stream_tts(text, tts_model_name="tts_models/en/ljspeech/tacotron2-DDC", speaker=None):
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

if __name__ == "__main__":
    text = """
    Hi, I am a Virtual Assistant made by Sjcet by Aibel Bin Zacariah
    """

    stream_tts(text)
