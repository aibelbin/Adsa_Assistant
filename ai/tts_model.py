from kokoro import KPipeline
import sounddevice as sd

pipeline = KPipeline(lang_code='b')
text = "Hello! This is a test of Kokoro running locally on your speakers."


generator = pipeline(text, voice='af_heart')
for i, (gs, ps, audio) in enumerate(generator):
    print(f"Segment {i} -> gs={gs}, ps={ps}")
    sd.play(audio, 24000)  # Play directly
    sd.wait()