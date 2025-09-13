from RealtimeSTT import AudioToTextRecorder


def responseText():
    recorder = AudioToTextRecorder()
    return recorder.text()

# if __name__ == '__main__':
#     # recorder = AudioToTextRecorder()

#     # while True:
#     #   print(responseText())