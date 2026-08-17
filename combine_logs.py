import os
import glob
import pandas as pd

DATA_DIR = "trade_data"
os.makedirs(DATA_DIR, exist_ok=True)

files = glob.glob(os.path.join(DATA_DIR, "backtest_trade_log_*.csv"))
if not files:
    print(f"No backtest_trade_log_*.csv files found in {DATA_DIR}/.")
else:
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, on_bad_lines="skip", engine="python")
            dfs.append(df)
            print(f"  Loaded {f}: {len(df)} rows")
        except Exception as e:
            print(f"  SKIPPED {f} — couldn't parse it at all ({e})")

    if not dfs:
        print("No files could be read.")
    else:
        combined = pd.concat(dfs, ignore_index=True)
        out_path = os.path.join(DATA_DIR, "backtest_trade_log.csv")
        combined.to_csv(out_path, index=False)
        print(f"\nCombined {len(dfs)} files into {len(combined)} total trades -> {out_path}")