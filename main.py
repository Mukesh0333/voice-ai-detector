from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from auth import verify_api_key
from model_inference import predict_voice
from utils import decode_base64_audio

app = FastAPI()

class AudioRequest(BaseModel):
    audio_base64: str


@app.post("/detect")
def detect_voice(request: AudioRequest, x_api_key: str = Header(None)):

    # API key check
    if not verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # Decode audio
    try:
        file_path = decode_base64_audio(request.audio_base64)
    except:
        raise HTTPException(status_code=400, detail="Invalid audio format")

    # TEMP placeholder result
    from model_inference import predict_voice
    result, confidence = predict_voice(file_path)


    return {
        "result": result,
        "confidence": confidence
    }
