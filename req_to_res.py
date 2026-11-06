import asyncio
import requests
from requests.exceptions import HTTPError, RequestException
from tts import enqueue_tts
import json
import os
from dotenv import load_dotenv
from pathlib import Path
from smart_home import control_device, DeviceNotFoundError
from playsound3 import playsound
from weather import get_weather
from text_chunker import Chunker, ThinkFilter
from sse_parse import parse_line, content_of, ToolCallAccumulator
import latency


load_dotenv(Path(__file__).parent / ".env")

MODEL = os.getenv("LLM_MODEL")
HOST = os.getenv("LLM_HOST")
CITY = os.getenv("CITY")
# Env overrides: LLM_MAX_TOKENS (reply cap, default 512), LLM_READ_TIMEOUT (seconds to wait on the model, default 300 so JIT model loads do not trip it), DEVICES_FILE
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))
READ_TIMEOUT = float(os.getenv("LLM_READ_TIMEOUT", "300"))
DEVICES_PATH = Path(os.getenv("DEVICES_FILE") or Path(__file__).parent / "devices.json")
MAX_CONTEXT = 40

with open(DEVICES_PATH) as f:
    config = json.load(f)

device_list = ""
for device_type, devices in config.items():
    if devices:
        device_list += f"\n{device_type.upper()}:\n"
        device_list += "\n".join(f'- "{d["name"]}"' for d in devices)
        device_list += "\n"

print(device_list)

SYSTEM_PROMPT = f"""
You are an AI voice assistant named Beemo who responds in a TTS manner back to the user. Keep messages informative but concise, and able to be understood by someone who is only listening to you.

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
                },
                "required": ["location"]
            }
        }
    }
]
context = []
followup_tools = ["get_weather"]
REQUIRED_ARGS = {"control_device": ("device", "action"), "get_weather": ("location",)}


class LLMError(Exception):
    """The model server answered with a non-200 status or an SSE error payload."""


# POST a streaming chat completion to LM Studio (optionally with tools); raises on network or HTTP errors.
def _post(messages, with_tools=False):
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": MAX_TOKENS,
        "stream": True,
    }
    if with_tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    latency.mark("llm_request_sent")
    res = requests.post(HOST, json=payload, stream=True, timeout=(5, READ_TIMEOUT))
    status = getattr(res, "status_code", 200)
    if status != 200:
        res.close()
        raise LLMError(f"LLM server returned HTTP {status}")
    return res


# Speak text via TTS; the only path to enqueue_tts, and it never raises (a TTS failure must not kill the turn).
def _speak(chunk):
    latency.mark("first_chunk_ready")
    try:
        enqueue_tts(chunk)
    except Exception as e:
        print("TTS Error: ", e)


# Read an SSE response, speaking text chunk by chunk as it streams; return (full reply, tool-call accumulator).
def _consume(res):
    chunker = Chunker()
    think = ThinkFilter()
    acc = ToolCallAccumulator()
    reply = ""

    # Feed visible (non-<think>) text to the reply and the chunker, speaking chunks as they complete.
    def emit(text):
        nonlocal reply
        if text:
            latency.mark("first_content_token")
            reply += text
            for chunk in chunker.feed(text):
                _speak(chunk)

    try:
        # chunk_size=None yields data as it arrives instead of waiting for 512 bytes
        for raw in res.iter_lines(chunk_size=None):
            kind, delta = parse_line(raw)
            if kind == "skip":
                continue
            if kind == "error":
                # On errors we raise and deliberately drop any half-built sentence still in the chunker
                raise LLMError(f"LLM stream error: {delta}")
            if kind == "done":
                break

            latency.mark("first_sse_chunk")
            text = content_of(delta)   # reasoning_content is deliberately ignored: never spoken
            latency.note_sse(bool(text))
            acc.add(delta)             # tool_calls may show up in any chunk, not just the first
            if delta.get("finish_reason") == "length":
                print(f"Warning: reply hit the max_tokens cap ({MAX_TOKENS}); raise LLM_MAX_TOKENS if it is cut off")
            emit(think.feed(text))
    finally:
        res.close()

    emit(think.flush())
    for chunk in chunker.flush():
        _speak(chunk)
    return reply, acc


# Post and consume one streaming request; return (reply, accumulator, error) and never raise on network/LLM failures.
def _stream(messages, with_tools=False):
    try:
        return (*_consume(_post(messages, with_tools)), None)
    except (RequestException, LLMError) as e:
        return "", ToolCallAccumulator(), e   # reply text is not kept on failure; send_to_model drops the turn


# Log a model failure and tell the user.
def _llm_failed(err):
    print(f"LLM error: {err!r}")
    _speak("I can't reach the model right now")


# Stream a reply to the given messages, speaking it as it arrives; return the text.
# with_tools=True sends the same tools list as the first request (keeps the prompt prefix identical for caching);
# any tool calls the model makes in this reply are ignored.
async def stream_and_speak(messages, with_tools=False):
    reply, acc, err = _stream(messages, with_tools)
    if err:
        _llm_failed(err)
    elif acc.all():
        print("Ignoring tool calls in the follow-up reply")
    return reply


# Keep context short without orphaning tool messages: it must always begin at a user message.
def trim_context(ctx, limit=MAX_CONTEXT):
    if len(ctx) > limit:
        ctx = ctx[len(ctx) - limit:]
    while ctx and ctx[0].get("role") != "user":
        ctx = ctx[1:]
    return ctx


# Format the weather tuple as the tool result text handed back to the model.
def _format_weather(data):
    description, temp, feels_like, max_temp, min_temp, wind_speed, precip = data
    return "\n".join([
        "All results are in imperial units.",
        f"Temperature: {temp}",
        f"Feels-like: {feels_like}",
        f"High: {max_temp}",
        f"Low: {min_temp}",
        f"Wind speed (mph): {wind_speed}",
        f"Precipitation (mm/h): {precip}",
        f"Description: {description}",
    ])


# Run the tool the model asked for; return (tool result text, whether a spoken follow-up is needed, summary suffix).
async def _run_tool(name, args):
    required = REQUIRED_ARGS.get(name)
    if required is None:
        _speak("I don't know how to do that")
        return f"Unknown tool {name or '(none)'}", False, ""
    if not isinstance(args, dict) or any(args.get(k) is None for k in required):
        raise KeyError(f"missing arguments for {name}: {args!r}")

    if name == "control_device":
        device, action = args["device"], args["action"]
        await control_device(device, action)
        try:
            playsound(Path(__file__).parent / "task-complete.mp3", block=False)
        except Exception as e:     # the chime is cosmetic; the action already succeeded
            print(f"Chime failed: {e}")
        return f"Successfully completed {name}", False, f" with arguments: device={device}, action={action}"

    # get_weather is blocking (requests), so keep it off the event loop
    data = await asyncio.to_thread(get_weather, args["location"], args.get("detailed", False))
    result = _format_weather(data)
    print(f"information received from weather api:\n{result}")
    return result, name in followup_tools, ""


# Execute one tool call and never raise: returns (tool result text, follow-up needed, summary text).
async def _execute(call):
    summary = f"Used Tool {call['name']}"
    try:
        result, followup, extra = await _run_tool(call["name"], json.loads(call["args"] or "{}"))
        return result, followup, summary + extra
    except DeviceNotFoundError:
        result = "Device not found amongst known devices"
    except HTTPError as e:
        status = e.response.status_code if e.response is not None else "unknown"
        print(f"Weather API HTTP error: {status}")
        result = "There was a problem reaching the API"
    except RequestException as e:
        print(f"Network error: {e}")
        result = "There was a problem reaching the API"
    except json.JSONDecodeError:
        result = "There was an error calling the tool"
    except ValueError as e:      # weather.get_geo: location not found
        print(f"Tool error: {e}")
        result = "I couldn't find that location"
    except Exception as e:       # offline plug, timeouts, bad args, anything else: still answer the tool call
        print(f"Tool error: {e!r}")
        result = "There was an error calling the tool"
    _speak(result)
    return result, False, summary


# Send the user's transcript to the model, speak the streamed reply, and handle any tool calls.
async def send_to_model(transcript):
    global context
    user_msg = {"role": "user", "content": transcript}
    context.append(user_msg)
    context = trim_context(context)

    reply, acc, err = _stream([{"role": "system", "content": SYSTEM_PROMPT}, *context], with_tools=True)
    if err:
        _llm_failed(err)
        context.pop()        # the turn was not answered; don't leave a dangling user message
        return reply

    calls = acc.all()
    if not calls:
        if reply.strip():
            context.append({"role": "assistant", "content": reply})
        else:
            # e.g. an unterminated <think> block used up max_tokens: avoid two user messages in a row
            _speak("Sorry, I don't have an answer for that")
            context.pop()
        return reply

    context.append({
        "role": "assistant",
        "content": reply or None,
        "tool_calls": [{
            "id": c["id"],
            "type": "function",
            "function": {"name": c["name"], "arguments": c["args"]}
        } for c in calls]
    })

    summaries, followup = [], False
    for call in calls:     # exactly one tool message per tool call, always
        result, needs_followup, summary = await _execute(call)
        context.append({"role": "tool", "tool_call_id": call["id"], "content": result})
        summaries.append(summary)
        followup = followup or needs_followup
    summary = "; ".join(summaries)
    if not followup:
        return summary

    # Same system prompt, same tools and same context prefix as the first request (rendered prompts embed the
    # tools, so omitting them would force the server to re-process the whole conversation)
    spoken = await stream_and_speak([{"role": "system", "content": SYSTEM_PROMPT}, *context], with_tools=True)
    if spoken.strip():
        context.append({"role": "assistant", "content": spoken})
    return summary + " and replied with:\n" + spoken
