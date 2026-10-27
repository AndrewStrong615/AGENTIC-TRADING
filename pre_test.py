import os
import sys

DATA_DIR = "trade_data"

PASS = "  [OK]  "
FAIL = "  [FAIL]"
WARN = "  [WARN]"

results = []


def check(label, ok, detail=""):
    level = PASS if ok else FAIL
    results.append((level, f"{label}{' — ' + detail if detail else ''}"))
    return ok


def warn(label, detail=""):
    results.append((WARN, f"{label}{' — ' + detail if detail else ''}"))


print("Running preflight checks...\n")

api_key = os.environ.get("ALPACA_API_KEY")
secret_key = os.environ.get("ALPACA_SECRET_KEY")
base_url = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2")

check("ALPACA_API_KEY is set", bool(api_key))
check("ALPACA_SECRET_KEY is set", bool(secret_key))
is_paper = "paper" in base_url
check("BASE_URL points to PAPER trading", is_paper, base_url)
if not is_paper:
    warn("BASE_URL does NOT look like a paper endpoint",
         "double-check this isn't pointed at a live-money account")

required_packages = {
    "alpaca_trade_api": "alpaca-trade-api",
    "sklearn": "scikit-learn",
    "vaderSentiment": "vaderSentiment",
    "hmmlearn": "hmmlearn",
    "joblib": "joblib",
    "pandas": "pandas",
    "numpy": "numpy",
}
for module_name, pip_name in required_packages.items():
    try:
        __import__(module_name)
        check(f"Package '{pip_name}' importable", True)
    except ImportError:
        check(f"Package '{pip_name}' importable", False, f"run: pip install {pip_name}")

for fname in ["Tradingcore.py", "meta_filter.py", "bot.py"]:
    check(f"{fname} present in this folder", os.path.exists(fname))

os.makedirs(DATA_DIR, exist_ok=True)
check(f"{DATA_DIR}/ folder exists (auto-created if missing)", os.path.isdir(DATA_DIR))

meta_model_path = os.path.join(DATA_DIR, "meta_model.joblib")
if os.path.exists(meta_model_path):
    check("Meta-model file found", True, meta_model_path)
else:
    warn("No meta-model found yet", "system will run with neutral (no-op) meta-filtering — this is fine, not an error")

weights_path = os.path.join(DATA_DIR, "expert_weights.json")
if os.path.exists(weights_path):
    check("Persisted expert weights found (learning carries over from before)", True, weights_path)
else:
    warn("No persisted expert weights yet", "first run will start all 4 experts at equal trust — expected on a first-ever run")

if api_key and secret_key:
    try:
        import alpaca_trade_api as tradeapi
        api = tradeapi.REST(api_key, secret_key, base_url, api_version="v2")
        account = api.get_account()
        check("Connected to Alpaca API", True)
        check("Account status is ACTIVE", account.status == "ACTIVE", f"status={account.status}")
        print(f"\n  Account equity: ${float(account.equity):,.2f}")
        print(f"  Buying power:   ${float(account.buying_power):,.2f}")
        pdt_flag = getattr(account, "pattern_day_trader", None)
        if pdt_flag is None:
            pdt_flag = getattr(account, "daytrade_count", "unavailable in this API version")
        print(f"  Pattern day trader flag: {pdt_flag}")
    except Exception as e:
        check("Connected to Alpaca API", False, str(e))

    try:
        clock = api.get_clock()
        if clock.is_open:
            print(f"\n  Market is currently OPEN.")
        else:
            print(f"\n  Market is currently CLOSED. Next open: {clock.next_open}")
    except Exception as e:
        warn("Couldn't fetch market clock", str(e))
else:
    check("Connected to Alpaca API", False, "skipped — API keys not set")

print(f"\n{'=' * 60}")
print("PREFLIGHT SUMMARY")
print(f"{'=' * 60}")
for level, msg in results:
    print(f"{level} {msg}")

n_fail = sum(1 for level, _ in results if level == FAIL)
n_warn = sum(1 for level, _ in results if level == WARN)

print(f"\n{n_fail} failure(s), {n_warn} warning(s).")
if n_fail == 0:
    print("\nLooks ready to go. Warnings above are informational, not blockers.")
    sys.exit(0)
else:
    print("\nFix the FAIL items above before starting trader_v3.py.")
    sys.exit(1)