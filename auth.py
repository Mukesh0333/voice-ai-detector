API_KEY = "impactx_api_aivoice_123"

def verify_api_key(key: str):
    if key != API_KEY:
        return False
    return True
