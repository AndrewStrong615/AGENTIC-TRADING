Autofix Automation — quick actions

This repo contains automation scripts that monitor backtests, reproduce incidents, triage issues, propose conservative fixes, and conditionally auto-merge safe changes.

If you want the agent's changes committed and pushed to your GitHub repo, run the provided PowerShell helper locally.

1. Commit & push (PowerShell)
   Open PowerShell in the repo root and run:

   .\scripts\commit_and_push.ps1 -Remote origin -Branch main

   The script will:
   - initialize git if necessary
   - configure a default user if not set
   - stage scripts, config, .github, backtest.py, bot.py
   - commit with a message including the Co-authored-by trailer
   - push to origin/main (ensure remote origin exists and you're authenticated)

2. If git push fails due to authentication:
   - Install Git: https://git-scm.com/downloads
   - Authenticate using GitHub CLI: gh auth login
   - Or use GitHub Desktop to push the changes

3. Add GH_TOKEN and (optional) ALPACA keys as GitHub repository secrets
   - GH_TOKEN: a Personal Access Token (repo scope) for PR creation/merge
   - ALPACA_API_KEY and ALPACA_SECRET_KEY: for real-market backtests

4. Run a local smoke test before scheduling
   - Create venv: python -m venv .venv
   - Activate: .\.venv\Scripts\Activate.ps1
   - Install deps: pip install -r requirements.txt (or pip install pandas numpy scikit-learn joblib vaderSentiment alpaca_trade_api yfinance matplotlib pytest pyyaml)
   - Run: python .\scripts\run_backtest.py
   - Run: python .\scripts\run_unit_tests.py
   - Run monitor once: .\scripts\run_monitor_now.ps1

5. Register scheduled Task Scheduler job (optional)
   - .\scripts\register_monitor_task.ps1 -IntervalMinutes 30 -TaskName "TradingBotMonitor" -PythonPath "C:\Path\To\python.exe"

If you prefer, instead of committing locally you can apply a patch I can generate; ask for "create patch" and I will produce one.
