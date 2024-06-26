# server.py
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from pydub import AudioSegment
import eventlet
import io
import time
import webrtcvad
import ssl
from pyogg import OpusFile
import wave

import logging
import torch
import torchaudio

import tempfile
import subprocess
from langchain.llms.ollama import Ollama

device = torch.device('cpu')  # gpu also can be used
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True)
import torch
from TTS.api import TTS
import os
os.environ["OPENAI_API_KEY"] = "abcd"
os.environ["OPENAI_API_BASE"] = "http://192.168.50.207:11434/"
chat_model = Ollama(

    #model="gpt-4-1106-preview",
    # #set base url for local
    # openai_api_base="http://192.168.50.207:11434/",#
    # openai_api_key="sk-1234",
    model="llama3:70b-instruct-q5_K_M",
    temperature=1,
    #max_tokens=8000,
 
    #default_query={"test": "test"},
)
# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(TTS().list_models().list_models())
 # Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils
print("Server is running")
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
socketio = SocketIO(app)

audio_buffer = io.BytesIO()
vad = webrtcvad.Vad()
from faster_whisper import WhisperModel

model_size = "distil-large-v3"

model = WhisperModel(model_size, device="cuda", compute_type="float16")

logging.basicConfig()
logging.getLogger("faster_whisper").setLevel(logging.DEBUG)
# Set aggressiveness mode, which is an integer between 0 and 3. 
# 0 is the least aggressive about filtering out non-speech, 3 is the most aggressive.
vad.set_mode(1)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('stopspeaking')
def handle_stopspeak(data):
    print('Received stopspeaking data')

@socketio.on('startspeaking')
def handle_startspeak(data):
    print('Received startspeaking data')

@socketio.on('playServerAudioStart')
def handle_playServerAudioStart(data):
    print('Received playServerAudioStart data')

@socketio.on('playServerAudioEnd')
def handle_playServerAudioEnd(data):
    print('Received playServerAudioEnd data')

@socketio.on('audio')
def handle_audio(data):
    print('Received audio data')
    global audio_buffer
    data = bytes(data)

    audio_buffer.write(data)
    if audio_buffer.tell() > 0:
        # Seek to the start of the BytesIO object before reading it
        audio_buffer.seek(0)
        # Convert the BytesIO object to bytes and create a wave file object
        wave_file = wave.open(audio_buffer, 'rb')
        # Read the audio data
        audio_data = wave_file.readframes(wave_file.getnframes())
        # Save the audio data to a temporary file
        with tempfile.NamedTemporaryFile(delete=True) as temp:
            temp.write(data)
            temp.flush()
            # # Read the sampling rate of the audio data
            # SAMPLING_RATE = 16000
            # # Use the Silero VAD to check if the audio data contains speech
            # wav = read_audio(temp.name,sampling_rate=SAMPLING_RATE)
            # speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=SAMPLING_RATE)
            # print(speech_timestamps)
            # # Run poetry run whisper-ctranslate2 --live_transcribe True --language en --speaker_name SPEAKER --hf_token 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 to transcribe the audio data
            # # If the audio data contains speech, transcribe it
            # if speech_timestamps:
            print('Speech detected')
            
            # Save the audio data to a temporary file
            
            segments, info = model.transcribe(temp.name, beam_size=5, language="en", condition_on_previous_text=False,vad_filter=True,vad_parameters={"threshold": .3})

            for segment in segments:
                print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
                res=chat_model.generate(prompts=[segment.text])
                print(res)
                res=res.generations[0][0].text
                print(res)
                render_tts(res,temp.name)
                with open("tts.wav", "rb") as f2:
                    print("sent audio to client")
                    emit('audioPlay', f2.read())

                # with open('useroutput.wav', 'wb') as sf:
                #     sf.write(data)
                #     sf.flush()
                #     # Run poetry run whisper
                #     command = ["whisper-ctranslate2", sf.name, "--language", "en","--device","cuda", "--speaker_name", "SPEAKER", "--hf_token", "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", "--verbose", "True"]
                #     subprocess.run(command, check=True, stdout=subprocess.PIPE)
                #     # Read and print the content of sample.txt
                #     with open(f'useroutput.txt', 'r') as f:
                #         print(f.read())
                        # with open("useroutput.wav", "rb") as f2:
                        # emit('audio', f2.read())
                    # emit('audio', data)
                    

def render_tts(text,voice="Luminary.mp3"):

   
    # # Run TTS
    # # ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
    # # Text to speech list of amplitude values as output
    # wav = tts.tts(text="Hello world!", speaker_wav="sample.mp3", language="en")


    # Text to speech to a file
    tts.tts_to_file(text=text, speaker_wav=voice, language="en", file_path="tts.wav")
    # Echo the audio data back to the client
    # read output.wav file

   
if __name__ == '__main__':
  # Create an SSL context
    socketio.run(app, log_output=True, host="0.0.0.0", port=5000, debug=True, use_reloader=False, keyfile='key.pem', certfile='cert.pem')