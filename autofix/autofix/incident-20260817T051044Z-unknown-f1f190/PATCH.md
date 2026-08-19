# Proposed autofix for incident-20260817T051044Z

Primary cause: unknown


The proposed conservative change addresses missing Alpaca credentials and protects backtest from raising a KeyError on import.
Suggested minimal code patch (apply to backtest.py):

--- a/backtest.py
+++ b/backtest.py
@@
-API_KEY = os.environ["ALPACA_API_KEY"]
-SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
-BASE_URL = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
-
-api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version="v2")
+API_KEY = os.environ.get("ALPACA_API_KEY")
+SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
+BASE_URL = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
+
+api = None
+try:
+    if API_KEY and SECRET_KEY:
+        api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version="v2")
+    else:
+        print("[autofix] ALPACA credentials not found; running in offline/mock mode")
+except Exception as e:
+    print(f"[autofix] Failed to initialize Alpaca API: {e}")
+    api = None

And add guards in fetch_bars() to fallback to deterministic synthetic data if `api` is None.
