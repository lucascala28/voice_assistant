import asyncio
from transcribe import collect_and_transcribe, warmup as warmup_stt
from req_to_res import send_to_model
from tts import start_tts, warmup as warmup_tts
import latency

# Warm up the models, then answer every transcribed request in a loop.
async def main():
    start_tts()
    print("Warming up models ...")
    warmup_tts()
    warmup_stt()
    for transcript in collect_and_transcribe():
        print(40*'-')
        print(f"Detected request:\n{transcript}\n")
        res = await send_to_model(transcript)
        print(f"Model output:\n{res}")
        latency.report()
        print(40*'-'+'\n')

if __name__ == "__main__":
    asyncio.run(main())