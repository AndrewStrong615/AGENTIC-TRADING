"""Conservative strategy optimizer.

Uses the current meta-model as a gate, searches a deliberately small parameter
space, evaluates on two newer holdout periods, and only promotes a new config
when it is materially better than the current saved config and not meaningfully
worse on either holdout.
"""
from __future__ import annotations
import argparse, csv, json, math, os, random, subprocess, sys, time, uuid
from pathlib import Path

BASE_DIR=Path(__file__).resolve().parent
BACKTEST=BASE_DIR/"backtest.py"
DATA_DIR=BASE_DIR/"trade_data"
RUNS_DIR=DATA_DIR/"backtests"
RESULTS_PATH=DATA_DIR/"optimizer_results.csv"
VALIDATION_PATH=DATA_DIR/"optimizer_validation.csv"
BEST_PATH=DATA_DIR/"best_strategy_config.json"

CONFIG_KEYS=("BAR_TIMEFRAME","MIN_SIGNAL_CONFIDENCE","MIN_META_WIN_PROB","MIN_ATR_PCT","MAX_ATR_PCT","MIN_TREND_CHANGE_PCT","MIN_BARS_BETWEEN_TRADES","RISK_PER_TRADE_PCT","MAX_POSITION_PCT","STOP_LOSS_PCT","TAKE_PROFIT_PCT","TRAIL_STOP_PCT","ATR_STOP_MULTIPLIER")

def f(v,d=0.0):
    try:return float(v)
    except:return float(d)

def metrics(summary):
    rows=summary.get("windows") or []
    if not rows:return {}
    def avg(k):
        vals=[f(r.get(k),0) for r in rows]
        return sum(vals)/len(vals) if vals else 0.0
    ret,dd,sh,so,cal,pf,n=map(avg,["total_return_pct_mean","max_drawdown_pct_mean","sharpe_mean","sortino_mean","calmar_mean","profit_factor_mean","num_trades_mean"])
    score=1.0*sh+0.4*so+0.2*cal+0.06*ret+0.12*pf-0.10*abs(dd)
    return {"return":ret,"drawdown":dd,"sharpe":sh,"sortino":so,"calmar":cal,"profit_factor":pf,"trades":n,"score":score}

def configs(n,seed=42):
    rng=random.Random(seed)
    base=[
      {"BAR_TIMEFRAME":"5m","MIN_SIGNAL_CONFIDENCE":0.35,"MIN_META_WIN_PROB":0.60,"RISK_PER_TRADE_PCT":0.003,"MAX_POSITION_PCT":0.15,"STOP_LOSS_PCT":0.8,"TAKE_PROFIT_PCT":2.0,"TRAIL_STOP_PCT":0.0,"ATR_STOP_MULTIPLIER":1.0,"MIN_ATR_PCT":0.05,"MAX_ATR_PCT":2.5,"MIN_TREND_CHANGE_PCT":0.0,"MIN_BARS_BETWEEN_TRADES":0},
      {"BAR_TIMEFRAME":"15m","MIN_SIGNAL_CONFIDENCE":0.40,"MIN_META_WIN_PROB":0.62,"RISK_PER_TRADE_PCT":0.003,"MAX_POSITION_PCT":0.12,"STOP_LOSS_PCT":1.0,"TAKE_PROFIT_PCT":2.5,"TRAIL_STOP_PCT":0.6,"ATR_STOP_MULTIPLIER":1.0,"MIN_ATR_PCT":0.05,"MAX_ATR_PCT":2.5,"MIN_TREND_CHANGE_PCT":0.0,"MIN_BARS_BETWEEN_TRADES":1},
      {"BAR_TIMEFRAME":"15m","MIN_SIGNAL_CONFIDENCE":0.45,"MIN_META_WIN_PROB":0.65,"RISK_PER_TRADE_PCT":0.0025,"MAX_POSITION_PCT":0.10,"STOP_LOSS_PCT":1.0,"TAKE_PROFIT_PCT":2.5,"TRAIL_STOP_PCT":0.8,"ATR_STOP_MULTIPLIER":1.0,"MIN_ATR_PCT":0.05,"MAX_ATR_PCT":2.0,"MIN_TREND_CHANGE_PCT":0.0,"MIN_BARS_BETWEEN_TRADES":2},
    ]
    seen={json.dumps(x,sort_keys=True) for x in base}
    while len(base)<n:
        stop=rng.choice([0.8,1.0,1.2]); tp=stop*rng.choice([2.0,2.5,3.0])
        c={"BAR_TIMEFRAME":rng.choice(["5m","15m"]),"MIN_SIGNAL_CONFIDENCE":rng.choice([0.35,0.4,0.45,0.5]),"MIN_META_WIN_PROB":rng.choice([0.6,0.62,0.65,0.68]),"RISK_PER_TRADE_PCT":rng.choice([0.0025,0.003,0.004]),"MAX_POSITION_PCT":rng.choice([0.10,0.12,0.15]),"STOP_LOSS_PCT":stop,"TAKE_PROFIT_PCT":round(tp,2),"TRAIL_STOP_PCT":rng.choice([0,0.6,0.8]),"ATR_STOP_MULTIPLIER":rng.choice([0.5,1.0,1.25]),"MIN_ATR_PCT":rng.choice([0,0.05,0.1]),"MAX_ATR_PCT":rng.choice([1.5,2.0,2.5]),"MIN_TREND_CHANGE_PCT":rng.choice([0,0.02]),"MIN_BARS_BETWEEN_TRADES":rng.choice([0,1,2])}
        key=json.dumps(c,sort_keys=True)
        if key not in seen:base.append(c);seen.add(key)
    return base[:n]

def run(cfg,offset,wins,days,symbols,tag):
    run_id=f"opt-{tag}-{uuid.uuid4().hex[:8]}"
    env=os.environ.copy(); env.update({k:str(v) for k,v in cfg.items()}); env.update({"BACKTEST_RUN_ID":run_id,"BACKTEST_END_OFFSET_DAYS":str(offset),"NUM_WINDOWS":str(wins),"WINDOW_LENGTH_DAYS":str(days),"ITERATIONS_PER_WINDOW":"1","BACKTEST_SYMBOLS":",".join(symbols),"SLEEP_BETWEEN_SYMBOLS":"0","NTFY_TOPIC":""})
    cp=subprocess.run([sys.executable,str(BACKTEST)],cwd=str(BASE_DIR),env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,check=False)
    sp=RUNS_DIR/run_id/"summary.json"; summary={}
    if sp.exists():
        try:summary=json.loads(sp.read_text())
        except:pass
    m=metrics(summary); m.update({"run_id":run_id,"returncode":cp.returncode,"output_tail":cp.stdout[-800:]}); return m

def meets_candidate(vs):
    # Conservative: every holdout must have non-negative return and acceptable drawdown.
    return all(v["return"] > 0 and v["drawdown"] > -15 and v["trades"] >= 20 for v in vs) and min(v["sharpe"] for v in vs) > 0.3

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--trials',type=int,default=8); ap.add_argument('--train-offset',type=int,default=240); ap.add_argument('--validation-offsets',default='0,30'); ap.add_argument('--train-windows',type=int,default=4); ap.add_argument('--validation-windows',type=int,default=3); ap.add_argument('--window-days',type=int,default=30); ap.add_argument('--symbols',default='AAPL,MSFT,NVDA,SPY'); ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args(); DATA_DIR.mkdir(exist_ok=True); syms=[x.strip().upper() for x in a.symbols.split(',') if x.strip()]; cfgs=configs(max(1,a.trials),a.seed)
    rows=[]
    for i,c in enumerate(cfgs,1):
        m=run(c,a.train_offset,a.train_windows,a.window_days,syms,f'train-{i}'); rows.append({**c,**{k:v for k,v in m.items() if k!='output_tail'}}); print(f"train {i}/{len(cfgs)} score={m.get('score',0):.2f} ret={m.get('return',0):+.2f}% sharpe={m.get('sharpe',0):.2f} dd={m.get('drawdown',0):.2f}%")
    rows.sort(key=lambda r:f(r.get('score')),reverse=True); top=rows[:min(3,len(rows))]
    fieldnames = sorted(set().union(*(r.keys() for r in rows), CONFIG_KEYS))
    with RESULTS_PATH.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    vrows=[]; offsets=[int(x) for x in a.validation_offsets.split(',') if x.strip()]
    for r in top:
        c={k:r[k] for k in CONFIG_KEYS}; allv=[]
        for off in offsets:
            m=run(c,off,a.validation_windows,a.window_days,syms,f'val-{off}'); allv.append(m); vrows.append({**c,"validation_offset":off,**{k:v for k,v in m.items() if k!='output_tail'}})
        r["validation_score"] = min(v["score"] for v in allv) if allv else -999
        r["validation_ok"] = meets_candidate(allv)
    with VALIDATION_PATH.open('w',newline='',encoding='utf-8') as fh:
        fields=sorted({k for r in vrows for k in r}); csv.DictWriter(fh,fieldnames=fields).writeheader(); csv.DictWriter(fh,fieldnames=fields).writerows(vrows)
    current={}
    if BEST_PATH.exists():
        try: current=json.loads(BEST_PATH.read_text()).get('config',{})
        except: current={}
    eligible=[r for r in top if r.get('validation_ok')]
    best=max(eligible,key=lambda r:f(r.get('validation_score'))) if eligible else None
    current_score=-999
    if current:
        cv=[]
        for off in offsets:
            cv.append(run(current,off,a.validation_windows,a.window_days,syms,f'current-{off}'))
        if cv: current_score=min(x['score'] for x in cv)
    promote=bool(best) and (not current or f(best.get('validation_score')) >= current_score + 0.10)
    if promote:
        payload={"mode":"conservative","config":{k:best[k] for k in CONFIG_KEYS},"validation_score":f(best.get('validation_score')),"promoted_at":time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),"requirements":{"multiple_holdouts":True,"score_margin":0.10,"min_sharpe":0.3,"min_return_each_holdout":0.0}}
        BEST_PATH.write_text(json.dumps(payload,indent=2),encoding='utf-8'); print('\nPROMOTED conservative config ->',BEST_PATH)
    else:
        print('\nKEPT current strategy config; no candidate cleared conservative promotion gates.')

if __name__=='__main__': main()
