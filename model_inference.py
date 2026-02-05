import torch
import joblib
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model

device = torch.device("cpu")

processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base-960h"
)

wav2vec = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base-960h"
).to(device)

wav2vec.eval()
torch.set_num_threads(1)

clf = joblib.load("models/voice_classifier.pkl")

def predict_voice(audio_path):
    audio, _ = librosa.load(audio_path, sr=16000)
    inputs = processor(audio, return_tensors="pt", sampling_rate=16000)

    with torch.no_grad():
        embedding = wav2vec(inputs.input_values).last_hidden_state.mean(dim=1)

    probs = clf.predict_proba(embedding.numpy())[0]

    label = "AI_GENERATED" if probs[1] > probs[0] else "HUMAN"
    confidence = float(max(probs))

    return label, confidence
