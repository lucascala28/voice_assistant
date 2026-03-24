from transcribe import collect_and_transcribe

for transcript in collect_and_transcribe():
    print(40*'-')
    print(transcript)
    print(40*'-'+'\n')