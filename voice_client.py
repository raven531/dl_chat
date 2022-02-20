import os
import io
import time
import subprocess
import requests

import sounddevice as sd

from scipy.io.wavfile import write
from google.cloud import speech

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""

fs = 44100
seconds = 2

record_path = os.path.join(os.getcwd(), "records")
record_file_path = record_path + "\\" + "output.wav"

client = speech.SpeechClient()

bot_name = "Sam"

if __name__ == "__main__":
    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    print('Starting: Speak now!!')
    sd.wait()
    print('Finish!!')
    write(record_file_path, fs, recording)

    time.sleep(0.5)

    """
    audio file format issue, transfer wav to flac by `sox` tool
    https://cloud.google.com/speech-to-text/docs/troubleshooting
    """
    subprocess.check_output(
        ['C:\\Program Files (x86)\\sox-14-4-2\\sox.exe', record_file_path,
         '--channel=1', '--bits=16', os.path.join(record_path, 'my_file.flac')])

    with io.open(os.path.join(record_path, 'my_file.flac'), 'rb') as audio_file:
        content = audio_file.read()
        audio = speech.RecognitionAudio(content=content)

    config = speech.RecognitionConfig(
        audio_channel_count=1,
        language_code="zh-TW"
    )

    response = client.recognize(request={"config": config, "audio": audio})

    if response is not None:

        for resp in response.results:
            msg = resp.alternatives[0].transcript
            print("you say: ", msg)

            result = requests.post(url="http://192.168.42.192:8080/", data={"sentence": msg})  # local
            # result = requests.post(url="", data={"sentence": msg})  # gcloud
            json_text = result.json()

            print(f'{bot_name}: {json_text.get("message")}')
