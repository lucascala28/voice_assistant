import requests
from tts import speak
import json
import os
from dotenv import load_dotenv
from pathlib import Path
from smart_home import control_device, DeviceNotFoundError
from playsound3 import playsound


load_dotenv(Path(__file__).parent / ".env")

MODEL = os.getenv("LLM_MODEL")
HOST = os.getenv("LLM_HOST")

with open(Path(__file__).parent / "devices.json") as f:
    config = json.load(f)

device_list = ""
for device_type, devices in config.items():
    if devices:
        device_list += f"\n{device_type.upper()}:\n"
        device_list += "\n".join(f'- "{d["name"]}"' for d in devices)
        device_list += "\n"

SYSTEM_PROMPT = """
You are an AI voice assistant named Jarvis who responds in a TTS manner back to the user. Keep messages informative but concise, and able to be understood by someone who is only listening to you.
You also are in control of smart devices and are tasked to control them upon user request. When a user asks you to control a smart device, respond ONLY with a JSON object in this format and nothing else:
{{"tool": "smartHome", "device_type": "<plugs|bulbs|tvs>", "device": "<device_name>", "action": "<action>"}}


Available Devices:
{device_list}

Actions per type of device:
    - PLUGS: on, off

Map informal naming to closest match. If not a device in the control request, respond normally and informatively.
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
        HOST,
        json={
            "model": MODEL,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *context],
            "temperature": 0.7,
            "stream": True
        },
        stream=True
    )
    reply = ""
    sentence_buffer = ""
    is_tool_call = False
    for line in res.iter_lines():
        if line:
            line = line.decode("utf-8")
            if line.startswith("data: "):
                line = line[6:]  # strip the "data: " prefix
            if line == "[DONE]":
                if is_tool_call:
                    try:
                        cleaned = reply.strip().replace("{{", "{").replace("}}", "}")
                        payload = json.loads(cleaned)
                        tool = payload["tool"]
                        
                        if tool == "smartHome":
                            device = payload["device"]
                            action = payload["action"]
                            dev_type = payload["device_type"]
                            
                            control_device(dev_type, device, action)

                            playsound(Path(__file__).parent / 'task-complete.mp3')
                               
                    except (DeviceNotFoundError) as e:
                        speak(f"Device {device} not found amongst known devices")

                    except (json.JSONDecodeError, KeyError) as e:
                        speak("There was an error calling the tool")
                elif sentence_buffer.strip():
                    speak(sentence_buffer)
                break
            try:
                chunk = json.loads(line)
                token = chunk["choices"][0]["delta"].get("content", "")
                reply += token
                sentence_buffer += token

                if reply.strip().startswith("{"):
                    is_tool_call = True
                    continue

                if sentence_buffer.endswith((".", "?", "!")):
                    speak(sentence_buffer.strip())
                    sentence_buffer = ""
            except Exception as e:
                print("Stream Error: ", e)
    context.append(
        {"role": "assistant", "content": reply}
    )
    return reply


