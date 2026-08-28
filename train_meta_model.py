from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, roc_auc_score, brier_score_loss

BASE_DIR=Path(__file__).resolve().parent
DATA_DIR=BASE_DIR/'trade_data'
LIVE_LOG=DATA_DIR/'trade_log.csv'
LEGACY_LOG=DATA_DIR/'backtest_trade_log.csv'
ARCHIVE_DIR=DATA_DIR/'backtests'
MODEL_PATH=DATA_DIR/'meta_model.joblib'
INFO_PATH=DATA_DIR/'meta_model_info.json'
TRAINING_PATH=DATA_DIR/'meta_training_data.csv'
MIN_TRADES_DEFAULT=200
MIN_VALIDATION_TRADES=30
MIN_AUC=0.55
MIN_AUC_IMPROVEMENT=0.02
MAX_BRIER_DEGRADATION=0.015
NUMERIC=["entry_confidence","entry_trend_change_pct","entry_atr_pct","entry_news_sentiment","entry_batch_weight","entry_online_weight","entry_markov_weight","entry_meanrev_weight","entry_hmm_regime_prob", 'entry_market_return_pct', 'entry_qqq_return_pct', 'entry_market_volatility_proxy', 'entry_relative_strength_pct', 'entry_relative_volume_ratio', 'entry_volume_acceleration', 'entry_distance_20_high_pct', 'entry_distance_20_low_pct', 'entry_distance_50_high_pct', 'entry_distance_50_low_pct', 'entry_atr_change_pct', 'entry_realized_volatility_pct', 'entry_volatility_percentile', 'entry_minutes_since_open', 'entry_minutes_to_close', 'entry_intraday_return_pct', 'entry_distance_from_open_pct', 'entry_opening_range_breakout', 'entry_gap_pct', 'entry_cross_section_rank', 'entry_breadth_pct_up', 'entry_sector_relative_return_pct', 'entry_news_count', 'entry_news_sentiment_change', 'entry_news_sentiment_dispersion']
CATEGORICAL=["entry_volatility_regime","entry_trend_regime","entry_hmm_regime"]

def discover_files():
    files=[]
    for x in (LIVE_LOG,LEGACY_LOG):
        if x.exists(): files.append(x)
    if ARCHIVE_DIR.exists(): files += sorted(ARCHIVE_DIR.rglob('backtest_trade_log_*.csv'))
    out=[]; seen=set()
    for x in files:
        r=x.resolve()
        if r not in seen: seen.add(r); out.append(x)
    return out

def load_source(path):
    try: df=pd.read_csv(path)
    except Exception as e:
        print(f'WARNING: could not read {path}: {e}'); return pd.DataFrame()
    if df.empty: return df
    df=df.copy()
    df['_source_file']=str(path.relative_to(BASE_DIR))
    df['_data_source']='live' if path==LIVE_LOG else ('backtest_legacy' if path==LEGACY_LOG else 'backtest_archive')
    return df

def collect():
    files=discover_files(); frames=[load_source(f) for f in files]
    frames=[x for x in frames if not x.empty]
    if not frames: return pd.DataFrame(), files
    df=pd.concat(frames,ignore_index=True,sort=False)
    if 'pnl_pct' not in df.columns: return pd.DataFrame(), files
    df['pnl_pct']=pd.to_numeric(df['pnl_pct'],errors='coerce')
    df=df.dropna(subset=['pnl_pct']).copy()
    for c in NUMERIC:
        if c not in df.columns: df[c]=np.nan
    for c in CATEGORICAL:
        if c not in df.columns: df[c]='unknown'
    df['_timestamp']=pd.to_datetime(df.get('exit_timestamp'),errors='coerce',utc=True) if 'exit_timestamp' in df.columns else pd.NaT
    keycols=[c for c in ['exit_timestamp','symbol','entry_price','exit_price','pnl_pct','entry_confidence'] if c in df.columns]
    if keycols: df=df.drop_duplicates(subset=keycols,keep='first')
    if df['_timestamp'].notna().any(): df=df.sort_values(['_timestamp','_source_file'],kind='stable')
    return df.reset_index(drop=True),files

def features(df):
    x=df.copy()
    for c in NUMERIC:
        x[c]=pd.to_numeric(x[c],errors='coerce')
        med=x[c].median(); x[c]=x[c].fillna(0 if pd.isna(med) else med)
    for c in CATEGORICAL:
        x[c]=x[c].astype(str).replace({'nan':'unknown','None':'unknown'})
    x=pd.get_dummies(x,columns=CATEGORICAL,prefix=['vol','trend','hmm'],dtype=float)
    cats=[c for c in x.columns if c.startswith(('vol_','trend_','hmm_'))]
    cols=NUMERIC+cats
    return x[cols].replace([np.inf,-np.inf],np.nan).fillna(0.0),cols

def evaluate(model,X,y,label):
    pred=model.predict(X); probs=model.predict_proba(X)[:,list(model.classes_).index(1)]
    auc=roc_auc_score(y,probs) if y.nunique()==2 else 0.5
    brier=brier_score_loss(y,probs)
    acc=accuracy_score(y,pred); bal=balanced_accuracy_score(y,pred)
    base=max(float(y.mean()),1-float(y.mean()))
    print(f'\n=== {label} ===')
    print(classification_report(y,pred,target_names=['lost','won'],zero_division=0))
    print('Confusion matrix (rows=actual, cols=predicted):')
    print(confusion_matrix(y,pred))
    print(f'Accuracy: {acc*100:.2f}% | Balanced: {bal*100:.2f}% | AUC: {auc:.4f} | Brier: {brier:.4f} | Majority baseline: {base*100:.2f}%')
    return {'accuracy':float(acc),'balanced_accuracy':float(bal),'auc':float(auc),'brier':float(brier),'baseline_accuracy':float(base)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--min-trades',type=int,default=MIN_TRADES_DEFAULT); a=ap.parse_args(); DATA_DIR.mkdir(exist_ok=True)
    df,files=collect()
    print('=== Meta-model data collection ===')
    for f in files: print(' ',f)
    if not df.size: print('No completed trade data found.'); sys.exit(1)
    print(f'\nCompleted trades collected: {len(df)}')
    counts=df['_data_source'].value_counts().to_dict(); print(f"Live trades: {counts.get('live',0)}"); print(f"Legacy backtest trades: {counts.get('backtest_legacy',0)}"); print(f"Archived backtest trades: {counts.get('backtest_archive',0)}")
    df.to_csv(TRAINING_PATH,index=False); print(f'Combined training data saved to {TRAINING_PATH}')
    df['won']=(df['pnl_pct']>0).astype(int)
    if df['won'].nunique()<2: print('Need both wins and losses.'); sys.exit(0)
    X,cols=features(df); y=df['won'].reset_index(drop=True)
    if df['_timestamp'].notna().mean()>=0.8:
        order=df['_timestamp'].sort_values(kind='stable').index; X=X.loc[order].reset_index(drop=True); y=y.loc[order].reset_index(drop=True); df=df.loc[order].reset_index(drop=True)
    val_n=max(MIN_VALIDATION_TRADES,int(len(df)*0.25)); val_n=min(val_n,len(df)//2)
    if len(df)<max(a.min_trades,val_n*2) or val_n<MIN_VALIDATION_TRADES: print(f'Not enough data for conservative validation. Need at least {a.min_trades} completed trades.'); sys.exit(0)
    split=len(df)-val_n; Xtr,Xv=X.iloc[:split],X.iloc[split:]; ytr,yv=y.iloc[:split],y.iloc[split:]
    if ytr.nunique()<2 or yv.nunique()<2: print('Train/validation segment lacks both outcomes.'); sys.exit(0)
    print(f'Training trades: {len(Xtr)} | Validation trades: {len(Xv)}')
    model=RandomForestClassifier(n_estimators=300,max_depth=5,min_samples_leaf=3,random_state=42,class_weight='balanced',n_jobs=-1); model.fit(Xtr,ytr)
    m=evaluate(model,Xv,yv,'Time-ordered validation on newer trades')
    prev=None
    if INFO_PATH.exists():
        try: prev=json.loads(INFO_PATH.read_text()).get('validation_metrics')
        except Exception: prev=None
    reasons=[]
    if m['auc']<MIN_AUC: reasons.append(f"AUC {m['auc']:.4f} below {MIN_AUC:.2f}")
    if prev:
        if m['auc']<float(prev.get('auc',0))+MIN_AUC_IMPROVEMENT: reasons.append(f"AUC improvement below {MIN_AUC_IMPROVEMENT:.2f}")
        if m['brier']>float(prev.get('brier',1))+MAX_BRIER_DEGRADATION: reasons.append('Brier score materially worse than previous model')
    if len(df)<a.min_trades: reasons.append(f"only {len(df)} trades; minimum is {a.min_trades}")
    imps=pd.Series(model.feature_importances_,index=cols).sort_values(ascending=False); print('\n=== Feature importance ==='); print(imps.to_string())
    if reasons:
        print('\nKEPT current meta-model.'); [print(' -',r) for r in reasons]
        return
    joblib.dump({'model':model,'feature_cols':cols},MODEL_PATH)
    info={'trained_at':pd.Timestamp.utcnow().isoformat(),'training_rows':len(Xtr),'validation_rows':len(Xv),'total_completed_trades':len(df),'data_sources':counts,'feature_cols':cols,'validation_metrics':m,'previous_validation_metrics':prev,'promotion':{'promoted':True,'minimum_trades':a.min_trades,'min_auc':MIN_AUC,'min_auc_improvement':MIN_AUC_IMPROVEMENT,'max_brier_degradation':MAX_BRIER_DEGRADATION}}
    INFO_PATH.write_text(json.dumps(info,indent=2,default=str)); print(f'\nPROMOTED new meta-model -> {MODEL_PATH}')

if __name__=='__main__': main()