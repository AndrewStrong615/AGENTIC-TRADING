import os
import glob
import pandas as pd

DATA_DIR = "trade_data"
os.makedirs(DATA_DIR, exist_ok=True)

files = sorted(glob.glob(os.path.join(DATA_DIR, "backtest_trade_log_*.csv")))
if not files:
    print(f"No backtest_trade_log_*.csv files found in {DATA_DIR}/.")
else:
    dfs = []
    for f in files:
        try:
            with open(f, newline="") as fh:
                raw_data_lines = max(sum(1 for _ in fh) - 1, 0)
        except OSError as e:
            print(f"  SKIPPED {f} — couldn't open it ({e})")
            continue

        try:
            df = pd.read_csv(f, on_bad_lines="skip", engine="python")
        except pd.errors.EmptyDataError:
            print(f"  SKIPPED {f} — file has no header/data")
            continue
        except Exception as e:
            print(f"  SKIPPED {f} — couldn't parse it at all ({e})")
            continue

        dfs.append(df)
        dropped = raw_data_lines - len(df)
        if dropped > 0:
            print(f"  Loaded {f}: {len(df)} rows "
                  f"({dropped} row(s) dropped — malformed or column count didn't match the header, "
                  f"likely a schema change mid-file)")
        else:
            print(f"  Loaded {f}: {len(df)} rows")

    if not dfs:
        print("No files could be read.")
    else:
        combined = pd.concat(dfs, ignore_index=True)
        out_path = os.path.join(DATA_DIR, "backtest_trade_log.csv")
        combined.to_csv(out_path, index=False)
        print(f"\nCombined {len(dfs)} files into {len(combined)} total trades -> {out_path}")
