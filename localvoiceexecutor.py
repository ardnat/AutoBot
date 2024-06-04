import subprocess
#from langchainagent import LangChainAgent
import sys

#agent = LangChainAgent()

# Read piped in text
piped_text = sys.stdin.read()

# output=agent.run(piped_text)

# # execute cmd.sh 
# command = [
#     "python", "tortoise/do_tts.py",
#     "--output_path", "/results",
#     "--preset", "ultra_fast",
#     "--voice", "geralt",
#     "--text", "Time flies like an arrow; fruit flies like a bananna."
# ]
# process = subprocess.Popen(command, stdout=subprocess.PIPE)
# stdout, stderr = process.communicate()

# print(stdout)

import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
print(TTS().list_models())

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Run TTS
# ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
# Text to speech list of amplitude values as output
wav = tts.tts(text="Hello world!", speaker_wav="my/cloning/audio.wav", language="en")


# Text to speech to a file
tts.tts_to_file(text="Hello world!", speaker_wav="my/cloning/audio.wav", language="en", file_path="output.wav")
from pydub import AudioSegment
from pydub.playback import play

# Load wav file
sound = AudioSegment.from_wav("output.wav")

# Play the sound
play(sound)
# whisper-ctranslate2 --live_transcribe True --language en --live_volume_threshold .1