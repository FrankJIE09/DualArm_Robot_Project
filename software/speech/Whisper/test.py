import whisper

model = whisper.load_model("large")
result = model.transcribe("audio_input.wav")
print(result["text"])