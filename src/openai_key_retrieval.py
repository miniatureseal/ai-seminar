import json
import os
import configparser
from pyprojroot.here import here

"""
Helper script to retrieve the OpenAI key from the file system or from the config file,
because I prefer to save my API keys in my file system instead of in the code.
Either save your key in the config.ini file without quotes like this:

KEY=YOUR_OPEN_AI_KEY_HERE

or in a secret.json file in a folder named 
.secret in your home directory, like this:
{
    "open_ai": "YOUR_OPEN_AI_KEY_HERE"
}

If the key is not found in the config.ini file, the script will look for the secret.json file.
"""


def _get_keys(path):
    with open(path) as f:
        return json.load(f)


def get_openai_key():
    config = configparser.ConfigParser()
    config.read(here("config.ini"))
    KEY = config.get("OPENAI", "KEY")

    if KEY == "NA":
        full_path_api_key = os.path.expanduser("~/.secret/secret.json")
        keys = _get_keys(full_path_api_key)
        return keys["open_ai"]

    else:
        return KEY
