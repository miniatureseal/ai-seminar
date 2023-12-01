import json
import os
import configparser
from pyprojroot.here import here


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
