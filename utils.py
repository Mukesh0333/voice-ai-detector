import base64

def decode_base64_audio(base64_string, output_file="temp_audio.mp3"):
    audio_bytes = base64.b64decode(base64_string)
    with open(output_file, "wb") as f:
        f.write(audio_bytes)
    return output_file
