from kokoro import KPipeline
import sounddevice as sd
import numpy as np

# Pick the right VoiceMeeter device index from sd.query_devices()
device_index = 51  
device_info = sd.query_devices(device_index)
device_samplerate = int(device_info["default_samplerate"])  # usually 48000

pipeline = KPipeline(lang_code='b')
text = "Hello! This is a test of Kokoro running locally on your speakers."

generator = pipeline(text, voice='af_heart')
for i, (gs, ps, audio) in enumerate(generator):
    print(f"Segment {i} -> gs={gs}, ps={ps}")

    # If Kokoro outputs 24000Hz, resample to device rate
    if device_samplerate != 24000:
        audio = np.interp(
            np.linspace(0, len(audio), int(len(audio) * device_samplerate / 24000)),
            np.arange(len(audio)),
            audio
        ).astype(np.float32)

    # Play on VoiceMeeter input
    sd.play(audio, samplerate=device_samplerate, device=device_index)
    sd.wait()

print(sd.query_devices())
