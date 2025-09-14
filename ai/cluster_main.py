from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os 
from RealtimeSTT import AudioToTextRecorder
import asyncio 
from localgroq import sendmessage 
from kokoro import KPipeline
import torch
import sounddevice as sd
import numpy as np

recorder = AudioToTextRecorder()


pipeline = KPipeline(lang_code='b')
text = "Hello! This is a test of Kokoro running locally on your speakers."


generator = pipeline(text, voice='af_heart')
for i, (gs, ps, audio) in enumerate(generator):
    print(f"Segment {i} -> gs={gs}, ps={ps}")
    sd.play(audio, 24000)  # Play directly
    sd.wait()

