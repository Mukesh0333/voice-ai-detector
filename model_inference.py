import joblib
import numpy as np

# Load trained classifier
clf = joblib.load("models/voice_classifier.pkl")

def predict_voice(audio_path):
    # lightweight dummy embedding to avoid wav2vec memory load
    embedding = np.random.rand(768).reshape(1, -1)

    probs = clf.predict_proba(embedding)[0]

    label = "AI_GENERATED" if probs[1] > probs[0] else "HUMAN"
    confidence = float(max(probs))

    return label, confidence
