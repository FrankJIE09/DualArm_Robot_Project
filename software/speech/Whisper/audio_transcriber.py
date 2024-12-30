import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import soundfile as sf
import json
import os
import time

# Load Whisper model
device = "cuda:0" if torch.cuda.is_available() else "cpu"  # Choose device: GPU (if available) or CPU
processor = AutoProcessor.from_pretrained("models_sage/whisper-tiny-finetune/")  # Load audio processor
model = AutoModelForSpeechSeq2Seq.from_pretrained("models_sage/whisper-tiny-finetune/")  # Load speech model
model.to(device)  # Load model onto the specified device
infer_pipe = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
                      feature_extractor=processor.feature_extractor, device=device)  # Setup inference pipeline
generate_kwargs = {}
generate_kwargs["language"] = "chinese"
generate_kwargs["task"] = "transcribe"
generate_kwargs["num_beams"] = 1


def ensure_directory_exists(directory):
    """Ensure the specified directory exists; create it if not."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def transcribe_and_save_audio(file_path="audio_input.wav", save_directory="audio", json_filename="audio.json"):
    # Ensure the save directory exists
    ensure_directory_exists(save_directory)

    # Read the input audio file and transcribe it
    audio_data, sample_rate = sf.read(file_path)  # Read audio file
    duration = len(audio_data) / sample_rate  # Calculate audio duration

    result = infer_pipe(audio_data, return_timestamps=False,
                        generate_kwargs=generate_kwargs)  # Perform speech recognition
    transcription_text = result['text']  # Get transcribed text
    print(transcription_text)

    # Generate a timestamped audio output path
    timestamp = int(time.time())
    audio_output_path = os.path.join(save_directory, f"record_{timestamp}.wav")
    sf.write(audio_output_path, audio_data, sample_rate)  # Save audio file

    # Create the transcription data structure
    transcription_data = {
        "audio": {"path": audio_output_path},
        "sentence": transcription_text,
        "duration": round(duration, 2),
        "sentences": [
            {"start": 0, "end": round(duration, 2), "text": transcription_text}
        ]
    }

    # Append the new transcription data to the JSON file as a new line
    json_output_path = os.path.join(save_directory, json_filename)
    with open(json_output_path, "a") as f:
        f.write(json.dumps(transcription_data, ensure_ascii=False, separators=(',', ':')) + "\n")

    print(f"Audio saved to: {audio_output_path}")
    print(f"Transcription appended to: {json_output_path}")
    return transcription_text # Return transcribed text


if __name__ == "__main__":
    input_audio_path = "audio_input.wav"

    # Transcribe and save audio
    transcription = transcribe_and_save_audio(file_path=input_audio_path)
    print(f"Transcription: {transcription}")  # Print transcription result

    # Save the transcription result to a separate JSON file for another program
    with open("config/transcription.json", "w") as f:
        json.dump({"text": transcription}, f)  # Save transcription text as JSON
