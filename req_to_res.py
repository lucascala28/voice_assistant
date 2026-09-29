import requests
from requests.exceptions import HTTPError
from tts import speak
import json
import os
from dotenv import load_dotenv
from pathlib import Path
from smart_home import control_device, DeviceNotFoundError
from playsound3 import playsound
from weather import get_weather


load_dotenv(Path(__file__).parent / ".env")

MODEL = os.getenv("LLM_MODEL")
HOST = os.getenv("LLM_HOST")
CITY = os.getenv("CITY")

with open(Path(__file__).parent / "devices.json") as f:
    config = json.load(f)

device_list = ""
for device_type, devices in config.items():
    if devices:
        device_list += f"\n{device_type.upper()}:\n"
        device_list += "\n".join(f'- "{d["name"]}"' for d in devices)
        device_list += "\n"

print(device_list)

SYSTEM_PROMPT = f"""
You are an AI voice assistant named Jarvis who responds in a TTS manner back to the user. Keep messages informative but concise, and able to be understood by someone who is only listening to you.

Available devices:
{device_list}

When controlling devices, always use 'off' when the user says 'turn off', 'disable', 'stop', or similar. Always use 'on' when the user says 'turn on', 'enable', 'start', or similar.
"""

tools = [
    {
        "type": "function",
        "function": {
            "name": "control_device",
            "description": "Control user's smart devices",
            "parameters": {
                "type": "object",
                "properties": {
                    "device": {
                        "type": "string",
                        "description": "Name of the device that the user wants to perform an action on"
                    },
                    "action": {
                        "type": "string",
                        "enum": ["on", "off"],
                        "description": "The action the user wants performed on the device"
                    }
                },
                "required": ["device", "action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather of a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": f"Specified location user wants current weather data from. If the user does not provide a specific city, use {CITY} as the location"
                    },
                    "detailed": {
                        "type": "boolean",
                        "description": "whether the user is asking for a full detailed report or the standard recap"
                    }
                }
            },
            "required": ["location", "detailed"]
        }
    }
]
context = []
followup_tools = ["get_weather"]

async def stream_and_speak(message):
    res = requests.post(
        HOST,
        json={
            "model": MODEL,
            "messages": message,
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
                line = line[6:]
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

    return reply


async def send_to_model(transcript):
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
            "stream": True,
            "tools": tools,
            "tool_choice": "auto"
        },
        stream=True
    )

    reply = ""
    sentence_buffer = ""
    is_tool_call = None
    tool_call = {"id": "", "name": "", "args": ""}

    for line in res.iter_lines():
        if line:
            line = line.decode("utf-8")
            if line.startswith("data: "):
                line = line[6:]  # strip the "data: " prefix
            if line == "[DONE]":
                if is_tool_call:
                    
                    context.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": tool_call["id"],
                            "type": "function",
                            "function": {
                                "name": tool_call["name"],
                                "arguments": tool_call["args"]
                            }
                        }]
                    })

                    reply = f"Used Tool {tool_call['name']}"
                    try: 
                        name = tool_call["name"]
                        args = json.loads(tool_call["args"])

                        if name == "control_device":
                            device = args.get('device')
                            action = args.get('action')
                        
                            reply += f" with arguments: device={device}, action={action}"

                            await control_device(device, action)
                            playsound(Path(__file__).parent / 'task-complete.mp3')

                        if name == "get_weather":
                            location = args.get('location')
                            detailed = args.get('detailed')

                            description, temp, feels_like, max_temp, min_temp, wind_speed, precip = await get_weather(location, detailed)
                            result = f"""
                            All results are in imperial units.
                            Temperature: {temp}
                            Feels-like: {feels_like}
                            High: {max_temp}
                            Low: {min_temp}
                            Wind speed (mph): {wind_speed}
                            Precipitation (mm/h): {precip}
                            Description: {description}
                            """
                            print(f"information received from weather api:\n{result}")
                        
                        if name in followup_tools:
                            followup = [
                                *context,
                                {"role": "assistant", "content": None, "tool_calls": [{
                                    "id": tool_call["id"],
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": tool_call["args"]
                                    }
                                }]},
                                {"role": "tool", "tool_call_id": tool_call["id"], "content": str(result)}
                            ]
                            result = await stream_and_speak(followup)
                            context.append(
                                {"role": "tool", "tool_call_id": tool_call["id"], "content": result}
                            )
                            reply += "and replied with:\n" + result
                            return reply
                        else:
                            context.append({
                                "role": "tool",
                                "tool_call_id": tool_call["id"],
                                "content": f"Successfully completed {tool_call['name']}"
                            })
                            return reply

                    
                    except DeviceNotFoundError:
                        speak(f"Device not found amongst known devices")
                    except (json.JSONDecodeError, KeyError) as e:
                        speak("There was an error calling the tool")
                    except (HTTPError) as e:
                        print(f"Weather API HTTP error: {e.response.status_code}")
                        speak("There was a problem reaching the API")

                elif sentence_buffer.strip():
                    speak(sentence_buffer)
                break
            try:
                chunk = json.loads(line)
                delta = chunk["choices"][0]["delta"]

                if is_tool_call is None:
                    is_tool_call = "tool_calls" in delta
                    if not is_tool_call and "content" in delta:
                        preamble_buffer += delta.get("content", "")
                        continue

                
                if is_tool_call:
                    if "tool_calls" in delta:
                        tc = delta["tool_calls"][0]
                        tool_call["id"] = tc.get('id', tool_call["id"])
                        if "function" in tc:
                            tool_call["name"] = tc["function"].get("name", tool_call["name"])
                            tool_call["args"] += tc["function"].get("arguments", "")
                    continue

                token = delta.get("content", "")
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


