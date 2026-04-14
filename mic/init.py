from transcribe import collect_and_transcribe
from req_to_res import send_to_model

for transcript in collect_and_transcribe():
    print(40*'-')
    print(f"Detected request:\n{transcript}\n")
    res = send_to_model(transcript)
    print(f"Model output:\n{res}")
    print(40*'-'+'\n')