import json
import os
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "trade_data")
CREDENTIALS_PATH = os.path.join(DATA_DIR, "credentials.json")

_lock = threading.Lock()

_DEFAULTS = {
    "alpaca_api_key": "",
    "alpaca_secret_key": "",
    "alpaca_base_url": "https://paper-api.alpaca.markets",
}


def load_credentials():
    if not os.path.exists(CREDENTIALS_PATH):
        return dict(_DEFAULTS)
    try:
        with open(CREDENTIALS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        merged = dict(_DEFAULTS)
        if isinstance(data, dict):
            merged.update({k: v for k, v in data.items() if k in _DEFAULTS})
        return merged
    except Exception:
        return dict(_DEFAULTS)


def save_credentials(api_key=None, secret_key=None, base_url=None):
    os.makedirs(DATA_DIR, exist_ok=True)
    with _lock:
        current = load_credentials()
        if api_key is not None:
            current["alpaca_api_key"] = str(api_key).strip()
        if secret_key is not None:
            current["alpaca_secret_key"] = str(secret_key).strip()
        if base_url is not None:
            current["alpaca_base_url"] = str(base_url).strip().rstrip("/")
        tmp = CREDENTIALS_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)
        os.replace(tmp, CREDENTIALS_PATH)
        return current


def masked(value, keep=4):
    if not value:
        return ""
    if len(value) <= keep * 2:
        return "*" * len(value)
    return f"{value[:keep]}{'*' * (len(value) - keep * 2)}{value[-keep:]}"


def status():
    creds = load_credentials()
    return {
        "api_key_set": bool(creds["alpaca_api_key"]),
        "secret_key_set": bool(creds["alpaca_secret_key"]),
        "api_key_masked": masked(creds["alpaca_api_key"]),
        "secret_key_masked": masked(creds["alpaca_secret_key"]),
        "base_url": creds["alpaca_base_url"],
        "is_paper": "paper" in creds["alpaca_base_url"].lower(),
    }


def build_subprocess_env(base_env=None):
    env = dict(base_env if base_env is not None else os.environ)
    creds = load_credentials()
    if creds["alpaca_api_key"]:
        env["ALPACA_API_KEY"] = creds["alpaca_api_key"]
    if creds["alpaca_secret_key"]:
        env["ALPACA_SECRET_KEY"] = creds["alpaca_secret_key"]
    if creds["alpaca_base_url"]:
        env["ALPACA_BASE_URL"] = creds["alpaca_base_url"]
    return env
