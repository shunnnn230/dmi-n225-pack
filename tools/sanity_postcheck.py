import os, json, time, csv, sys, argparse, traceback
from pathlib import Path
import pandas as pd
import numpy as np

# 依存: yfinance, ta（無ければ最小限で動く）
try:
    import yfinance as yf
except Exception:
    yf = None
try:
    from ta.trend import ADXIndicator
except Exception:
    ADXIndicator = None

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT  = ROOT / "out" / "n225"
REPORT = OUT / "run_report.json"

SEED = [s.strip() for s in open(DATA/"n225_tickers.txt", "r", encoding="utf-8").read().splitlines() if s.strip()]

def _read_metrics(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def _need_fallback(df: pd.DataFrame, min_uniqs=200):
    if df.empty: return True
    cols = {c.lower() for c in df.columns}
    if "ticker" not in cols: return True
    # tradesが全部0、もしくはユニークティッカーが少なすぎる
    try:
        trades_sum = pd.to_numeric(df.get("trades"), errors="coerce").fillna(0).sum()
    except Exception:
        trades_sum = 0
    uniqs = df["ticker"].nunique()
    return (trades_sum <= 0) or (uniqs < min_uniqs)

def _pick_components(metrics_df):
    # 優先1: data/n225_tickers.txt（200以上あれば採用）
    if len(SEED) >= 200:
        src = "local_txt"
        return SEED, src
    # 優先2: 直前のmetricsから復元
    if not metrics_df.empty and metrics_df["ticker"].nunique() >= 200:
        src = "previous_metrics"
        return metrics_df["ticker"].dropna().astype(str).unique().tolist(), src
    # 最後: ローカルtxt（少数）でも絶対空にはしない
    src = "seed_minimal"
    return SEED, src

def _download_series(ticker):
    if yf is None: return None
    try:
        df = yf.download(ticker, period="5y", interval="1d", auto_adjust=True, progress=False, threads=False)
        if df is None or len(df) < 30: return None
        df = df.dropna()
        if {"Open","High","Low","Close"}.issubset(df.columns):
            return df
        return None
    except Exception:
        return None

def _calc_simple_metrics(df):
    # 期間単純リターンとCAGRのみ（フォールバック用）
    start = df.index[0]
    end   = df.index[-1]
    total = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1.0) * 100.0
    years = max(1e-9, (end - start).days / 365.25)
    cagr  = ((df["Close"].iloc[-1] / df["Close"].iloc[0]) ** (1/years) - 1.0) * 100.0
    return total, cagr, str(start.date()), str(end.date())

def _infer_signal(df):
    sig, reason = "HOLD", "NONE"
    if ADXIndicator is None:
        return sig, reason
    try:
        adx = ADXIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=14, fillna=False)
        s = adx.adx_pos() - adx.adx_neg()
        if len(s) >= 2:
            is_gc = (s.iloc[-2] <= 0) and (s.iloc[-1] > 0)
            is_dc = (s.iloc[-2] >= 0) and (s.iloc[-1] < 0)
            if is_gc: sig, reason = "BUY", "GC"
            elif is_dc: sig, reason = "SELL", "DC"
    except Exception:
        pass
    return sig, reason

def _write_outputs(rows, signals):
    # CSV
    met = OUT / "latest_metrics.csv"
    with open(met, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ticker","name_jp","mgc","mdc","trades","win_rate","total_return","cagr","max_dd","start","end","updated_at"])
        w.writeheader()
        for r in rows: w.writerow(r)
    # signals.csv
    import pandas as pd
    pd.DataFrame([dict(ticker=k, **v) for k,v in signals.items()]).to_csv(OUT/"latest_signals.csv", index=False)
    # pack.json
    top = sorted(
        [{"ticker": r["ticker"], "name_jp": r.get("name_jp") or r["ticker"], "total_pct": r["total_return"], "cagr_pct": r["cagr"]} for r in rows],
        key=lambda x: (x["total_pct"] if x["total_pct"] is not None else -1e9),
        reverse=True
    )
    pack = dict(updated_at=time.strftime("%Y-%m-%dT%H:%M:%S+09:00", time.localtime()),
                top=top, stars={}, signals=signals, metrics={})
    with open(OUT/"latest_pack.json","w",encoding="utf-8") as f:
        json.dump(pack, f, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="強制的にフォールバック生成")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    # 直前成果物
    metrics_path = OUT/"latest_metrics.csv"
    met_df = _read_metrics(metrics_path) if metrics_path.exists() else pd.DataFrame()
    need = args.force or _need_fallback(met_df, min_uniqs=200)

    report = dict(step="postcheck", force=args.force, need_fallback=bool(need),
                  reason="", component_source="", counts={})

    if not need:
        report["reason"] = "metrics OK (>=200 & trades>0)"
        with open(REPORT,"w",encoding="utf-8") as f: json.dump(report,f,ensure_ascii=False,indent=2)
        print("[OK] fallback不要")
        return

    # フォールバック
    tickers, src = _pick_components(met_df)
    report["component_source"] = src
    print(f"[INFO] fallback実行: source={src}, tickers={len(tickers)}")

    rows = []; signals = {}
    ok_cnt = 0; ng_cnt = 0
    for t in tickers:
        try:
            df = _download_series(t)
            if df is None:
                ng_cnt += 1
                continue
            tot, cagr, start, end = _calc_simple_metrics(df)
            sig, why = _infer_signal(df)
            rows.append(dict(
                ticker=t, name_jp="", mgc="", mdc="", trades=1, win_rate=np.nan,
                total_return=tot, cagr=cagr, max_dd=np.nan,
                start=start, end=end,
                updated_at=time.strftime("%Y-%m-%dT%H:%M:%S+09:00", time.localtime())
            ))
            signals[t] = dict(signal=sig, reason=why)
            ok_cnt += 1
        except Exception:
            ng_cnt += 1

    if ok_cnt == 0:
        report["reason"] = "fallbackも0件"
        with open(REPORT,"w",encoding="utf-8") as f: json.dump(report,f,ensure_ascii=False,indent=2)
        print("[FATAL] fallback生成0件")
        sys.exit(1)

    _write_outputs(rows, signals)
    report["reason"] = "fallback生成"
    report["counts"] = dict(tickers=len(tickers), ok=ok_cnt, ng=ng_cnt, rows=len(rows))
    with open(REPORT,"w",encoding="utf-8") as f: json.dump(report,f,ensure_ascii=False,indent=2)
    print(f"[DONE] fallback生成: ok={ok_cnt} ng={ng_cnt} rows={len(rows)}")

if __name__ == "__main__":
    main()
