from transcribe import collect_and_transcribe
import requests
from tts import speak
import json


SYSTEM_PROMPT = """
You are an AI voice assistant named mycroft who responds in a TTS manner back to the user. Keep messages informative but concise, and able to be understood by someone who is only listening to you.
"""
context = []
def send_to_model(transcript):
    global context
    context.append(
        {"role": "user", "content": transcript}
    )
    if len(context) > 40:
        context = context[2:]
    res = requests.post(
        "http://localhost:1234/v1/chat/completions",
        json={
            "model": "qwen2.5-14b-instruct-1m",
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *context],
            "temperature": 0.7,
            "stream": True
        },
        stream=True
    )
    reply = ""
    sentence_buffer = ""
    for line in res.iter_lines():
        if line:
            line = line.decode("utf-8")
            if line.startswith("data: "):
                line = line[6:]  # strip the "data: " prefix
            if line == "[DONE]":
                if sentence_buffer.strip():
                    speak(sentence_buffer)
                break
            try:
                chunk = json.loads(line)
                token = chunk["choices"][0]["delta"].get("content", "")
                reply += token
                sentence_buffer += token
                if sentence_buffer.endswith((".", "?", "!")):
                    speak(sentence_buffer.strip())
                    sentence_buffer = ""
            except Exception as e:
                print("Stream Error: ", e)
    context.append(
        {"role": "assistant", "content": reply}
    )
    return reply


