import subprocess
#from langchainagent import LangChainAgent
import sys

#agent = LangChainAgent()

# Read piped in text
#piped_text = sys.stdin.read()

# output=agent.run(piped_text)

# execute cmd.sh 
command = [
    "python", "tortoise/do_tts.py",
    "--output_path", "/results",
    "--preset", "ultra_fast",
    "--voice", "geralt",
    "--text", "Time flies like an arrow; fruit flies like a bananna."
]
process = subprocess.Popen(command, stdout=subprocess.PIPE)
stdout, stderr = process.communicate()

print(stdout)