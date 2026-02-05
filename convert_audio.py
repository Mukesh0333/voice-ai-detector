import os
import librosa
import soundfile as sf

INPUT_DIR = "dataset"
OUTPUT_DIR = "processed_dataset"
MAX_DURATION = 12  # seconds

def process_file(input_path, output_path):
    try:
        audio, sr = librosa.load(input_path, sr=16000, mono=True)

        # Trim long audio
        max_samples = MAX_DURATION * 16000
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, audio, 16000)

    except Exception as e:
        print("Error processing", input_path, e)

for root, _, files in os.walk(INPUT_DIR):
    for file in files:
        if file.lower().endswith((".wav", ".mp3", ".flac")):
            in_path = os.path.join(root, file)
            out_path = in_path.replace(INPUT_DIR, OUTPUT_DIR)
            out_path = out_path.rsplit(".", 1)[0] + ".wav"
            process_file(in_path, out_path)

print("✅ Audio preprocessing complete.")
