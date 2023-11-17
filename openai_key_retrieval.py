import json
import os

def _get_keys(path):
    with open(path) as f:
        return json.load(f)
    
def get_openai_key():
    full_path_api_key = os.path.expanduser("~/.secret/secret.json")
    keys = _get_keys(full_path_api_key)
    return keys["open_ai"]