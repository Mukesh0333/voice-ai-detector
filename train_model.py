import os
import torch
import librosa
import joblib
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.linear_model import LogisticRegression

DATASET_PATH = "small_dataset"
MODEL_OUT = "models/voice_classifier.pkl"

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec.eval()

X, y = [], []

def extract_features(path):
    audio, _ = librosa.load(path, sr=16000)
    inputs = processor(audio, return_tensors="pt", sampling_rate=16000)
    with torch.no_grad():
        emb = wav2vec(inputs.input_values).last_hidden_state.mean(dim=1)
    return emb.numpy().flatten()

# HUMAN = 0
# HUMAN
print("Processing HUMAN audio...")
for root, _, files in os.walk(os.path.join(DATASET_PATH, "human")):
    for f in files:
        if f.endswith(".wav"):
            path = os.path.join(root, f)
            print("Human:", path)
            X.append(extract_features(path))
            y.append(0)

# AI
print("Processing AI audio...")
for root, _, files in os.walk(os.path.join(DATASET_PATH, "ai")):
    for f in files:
        if f.endswith(".wav"):
            path = os.path.join(root, f)
            print("AI:", path)
            X.append(extract_features(path))
            y.append(1)


print("Training classifier...")
clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)

os.makedirs("models", exist_ok=True)
joblib.dump(clf, MODEL_OUT)

print("✅ Model trained and saved:", MODEL_OUT)
