from transcribe import collect_and_transcribe
import requests

def send_to_model(transcript):
    res = requests.post(
        "http://localhost:1234/v1/chat/completions",
        json={
            "model": "qwen2.5-14b-instruct-1m",
            "messages": [{"role": "user", "content": transcript}],
            "temperature": 0.7,
            "stream": True
        },
        stream=True
    )
    reply = ""
    for line in res.iter_lines():
        if line:
            line = line.decode("utf-8")
            if line.startswith("data: "):
                line = line[6:]  # strip the "data: " prefix
            if line == "[DONE]":
                break
            try:
                chunk = json.loads(line)
                token = chunk["choices"][0]["delta"].get("content", "")
                print(token, end="", flush=True)
                reply += token
            except:
                pass
    print()
    return reply


