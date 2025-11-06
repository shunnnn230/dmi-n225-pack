# filename: sony_mgc_mdc.py  (vFINAL)
# Goal: 直近5年・1日足・DMI(+DI/-DI)×RSI80利確
#   1) MGC（買い角度帯）→ 取引数下限を高めにして“0–60°”に乗りやすく
#   2) MDC（売り角度帯）→ 幅/位置に現実的な制約を入れて“50–65°”域に寄せる
#   3) 最終シミュレーション → CSV出力（UTF-8-SIG）

import argparse, os, sys, math, csv
import numpy as np
import pandas as pd
import yfinance as yf
from ta.trend import ADXIndicator
from ta.momentum import RSIIndicator

# --- 日本語社名（最低限の既定） ---
JP_NAME_MAP = {"6758.T": "ソニーグループ"}

def load_name_csv(path):
    m={}
    if path and os.path.exists(path):
        with open(path,"r",encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                t=(r.get("ticker") or r.get("code") or r.get("symbol") or "").strip()
                n=(r.get("name_jp") or r.get("name") or r.get("jp") or "").strip()
                if t and n: m[t]=n
    return m

# --- yfinanceの列を1次元OHLCに整形 ---
def _to_1d_series(x, idx, name):
    s = x.iloc[:,0] if isinstance(x, pd.DataFrame) else x
    arr = np.asarray(s)
    if arr.ndim>1: arr=arr.reshape(-1)
    return pd.Series(arr, index=idx, name=name)

def ensure_ohlc_1d(df):
    if isinstance(df.columns, pd.MultiIndex):
        # 0/1どちらかに Open/High/Low/Close が揃ってる方を採用
        for lvl in [0,1]:
            if set(["Open","High","Low","Close"]).issubset(df.columns.get_level_values(lvl)):
                df = df.droplevel(1-lvl, axis=1)
                break
    for c in ["Open","High","Low","Close"]:
        if c not in df.columns:
            raise ValueError(f"OHLC列 '{c}' が見つかりません。列構成: {list(df.columns)}")
        df[c] = pd.to_numeric(_to_1d_series(df[c], df.index, c), errors="coerce")
    return df.dropna(subset=["Open","High","Low","Close"])

# --- DMI(+DI/-DI), ADX, RSI, スプレッドS(+DI - -DI) ---
def indicators(df, dmi_win=14, rsi_win=14):
    adx = ADXIndicator(df["High"], df["Low"], df["Close"], window=dmi_win, fillna=False)
    di_pos = adx.adx_pos()
    di_neg = adx.adx_neg()
    adx_v  = adx.adx()
    rsi    = RSIIndicator(df["Close"], window=rsi_win, fillna=False).rsi()
    S      = di_pos - di_neg
    return di_pos, di_neg, adx_v, rsi, S

# --- 角度θ：Sの直近k本の傾き→arctan→deg ---
def theta_from_spread(S: pd.Series, k=5) -> pd.Series:
    th = np.full(len(S), np.nan)
    x  = np.arange(k)
    for i in range(k-1, len(S)):
        w = S.iloc[i-k+1:i+1].values
        if np.isnan(w).any(): continue
        m = np.polyfit(x, w, 1)[0]
        th[i] = np.degrees(np.arctan(m))
    return pd.Series(th, index=S.index, name="theta")

# --- OGC/DC 検出（符号反転）---
def detect_cross(S: pd.Series):
    prev = S.shift(1)
    ogc = ((prev <= 0) & (S > 0)).values.nonzero()[0].tolist()  # +DI が -DI を上抜け
    odc = ((prev >= 0) & (S < 0)).values.nonzero()[0].tolist()  # -DI が +DI を上抜け
    return ogc, odc

def in_band(a, band):
    return (band is not None) and (not np.isnan(a)) and (band[0] <= a <= band[1])

# --- 売買シミュレーション ---
def simulate(df, ogc_idx, odc_idx, theta, rsi, adx, *,
             mgc_band, mdc_band, rsi_exit=80, min_hold=8, adx_min=15, cost=0.002):
    holding=False; ei=None; eang=None
    trades=[]
    ogc_set, odc_set = set(ogc_idx), set(odc_idx)
    for t in range(1, len(df)):
        # Exit 優先
        if holding:
            held = t - ei
            # RSI利確（先に来たら追加売却しない）
            if (not np.isnan(rsi.iat[t])) and rsi.iat[t] >= rsi_exit:
                ret = df["Close"].iat[t]/df["Close"].iat[ei] - 1.0
                ret -= cost
                trades.append({
                    "entry_date": df.index[ei].date(),
                    "exit_date":  df.index[t].date(),
                    "entry_px":   float(df["Close"].iat[ei]),
                    "exit_px":    float(df["Close"].iat[t]),
                    "entry_angle":float(eang) if eang is not None else np.nan,
                    "exit_angle": np.nan,
                    "exit_reason":"RSI",
                    "ret":        float(ret),
                })
                holding=False; ei=None; eang=None
            else:
                # DCで売却（保有8日未満はNC扱い→スキップ）
                if (t in odc_set) and (held >= min_hold):
                    xang = abs(theta.iat[t]) if not np.isnan(theta.iat[t]) else np.nan
                    if (mdc_band is None) or in_band(xang, mdc_band):
                        ret = df["Close"].iat[t]/df["Close"].iat[ei] - 1.0
                        ret -= cost
                        trades.append({
                            "entry_date": df.index[ei].date(),
                            "exit_date":  df.index[t].date(),
                            "entry_px":   float(df["Close"].iat[ei]),
                            "exit_px":    float(df["Close"].iat[t]),
                            "entry_angle":float(eang) if eang is not None else np.nan,
                            "exit_angle": float(xang) if not np.isnan(xang) else np.nan,
                            "exit_reason":"DC",
                            "ret":        float(ret),
                        })
                        holding=False; ei=None; eang=None
        # Entry
        if (not holding) and (t in ogc_set):
            ang = theta.iat[t]
            if in_band(ang, mgc_band) and (not np.isnan(adx.iat[t])) and (adx.iat[t] >= adx_min):
                holding=True; ei=t; eang=float(ang) if not np.isnan(ang) else np.nan

    # メトリクス
    if not trades:
        return trades, dict(trade_count=0, win_rate=np.nan, total_return=0.0,
                            final_equity=1.0, cagr=np.nan, max_dd=np.nan,
                            start=None, end=None)
    eq=1.0; curve=[eq]
    for tr in trades:
        eq *= (1.0 + tr["ret"])
        curve.append(eq)
    curve = np.array(curve)
    start = pd.to_datetime(trades[0]["entry_date"])
    end   = pd.to_datetime(trades[-1]["exit_date"])
    years = max((end-start).days, 1)/365.25
    cagr  = (eq**(1/years)-1) if years>0 else np.nan
    wins  = sum(1 for tr in trades if tr["ret"]>0)
    wr    = wins/len(trades)
    peaks = np.maximum.accumulate(curve)
    dd    = (curve - peaks)/peaks
    mdd   = dd.min()
    return trades, dict(trade_count=len(trades), win_rate=float(wr),
                        total_return=float(eq-1.0), final_equity=float(eq),
                        cagr=float(cagr), max_dd=float(mdd),
                        start=str(start.date()), end=str(end.date()))

# --- MGC探索（mdcは(0,90)固定）---
def search_mgc(df, ogc, odc, th, rsi, adx, *, step, min_trades_mgc, min_hold, adx_min, cost):
    best=None
    for L in range(0, 91, step):
        for U in range(L, 91, step):
            mgc=(L,U)
            _,m = simulate(df, ogc, odc, th, rsi, adx,
                           mgc_band=mgc, mdc_band=(0,90),
                           rsi_exit=80, min_hold=min_hold, adx_min=adx_min, cost=cost)
            if m["trade_count"] < min_trades_mgc: 
                continue
            score = m["final_equity"]
            if (best is None) or (score>best["score"]) or (math.isclose(score,best["score"]) and m["trade_count"]>best["m"]["trade_count"]):
                best = dict(mgc=mgc, m=m, score=score)
    return best

# --- MDC探索（制約付き）---
def search_mdc(df, ogc, odc, th, rsi, adx, *, mgc_fixed, step, min_trades_mdc, min_hold, adx_min, cost,
               mdc_l_min, mdc_u_max, mdc_max_width):
    best=None
    for L in range(max(0,mdc_l_min), 91, step):
        for U in range(L, min(91, mdc_u_max+1), step):
            if (U-L) > mdc_max_width: 
                continue
            mdc=(L,U)
            _,m = simulate(df, ogc, odc, th, rsi, adx,
                           mgc_band=mgc_fixed, mdc_band=mdc,
                           rsi_exit=80, min_hold=min_hold, adx_min=adx_min, cost=cost)
            if m["trade_count"] < min_trades_mdc:
                continue
            score = m["final_equity"]
            # 同点なら 1) 取引数多い 2) 帯幅が狭い を優先
            width = U-L
            if (best is None) or (score>best["score"]) \
               or (math.isclose(score,best["score"]) and m["trade_count"]>best["m"]["trade_count"]) \
               or (math.isclose(score,best["score"]) and math.isclose(m["trade_count"],best["m"]["trade_count"]) and width < (best["mdc"][1]-best["mdc"][0])):
                best = dict(mdc=mdc, m=m, score=score)
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="6758.T")
    ap.add_argument("--period", default="5y")
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--dmi", type=int, default=14)
    ap.add_argument("--rsi", type=int, default=14)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--min-hold", type=int, default=8)
    # ★ここが“再現”のキモ：MGCは件数しきい値を高めに
    ap.add_argument("--min-trades-mgc", type=int, default=8)
    ap.add_argument("--min-trades-mdc", type=int, default=5)
    ap.add_argument("--adx-min", type=float, default=15)
    ap.add_argument("--rsi-exit", type=float, default=80)
    ap.add_argument("--cost", type=float, default=0.002)
    # 売り帯の制約（現実的な出口を優先）
    ap.add_argument("--mdc-l-min", type=int, default=45)
    ap.add_argument("--mdc-u-max", type=int, default=70)
    ap.add_argument("--mdc-max-width", type=int, default=20)
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--name-map", default="")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 価格取得
    df = yf.download(args.ticker, period=args.period, interval=args.interval,
                     auto_adjust=True, progress=False, group_by="column", threads=False)
    if df is None or len(df)==0:
        print("[ERROR] データ取得失敗"); sys.exit(1)
    df = ensure_ohlc_1d(df)

    # 指標
    di_p, di_n, adx, rsi, S = indicators(df, args.dmi, args.rsi)
    th = theta_from_spread(S, k=args.k)
    ogc, odc = detect_cross(S)

    # MGC探索
    mgc_res = search_mgc(df, ogc, odc, th, rsi, adx,
                         step=args.step, min_trades_mgc=args.min_trades_mgc,
                         min_hold=args.min_hold, adx_min=args.adx_min, cost=args.cost)
    if mgc_res is None:
        print("[WARN] 有効なMGCが見つかりませんでした。しきい値やADX等を緩めてください。")
        sys.exit(0)
    best_mgc = mgc_res["mgc"]

    # MDC探索（MGC固定）
    mdc_res = search_mdc(df, ogc, odc, th, rsi, adx,
                         mgc_fixed=best_mgc, step=args.step, min_trades_mdc=args.min_trades_mdc,
                         min_hold=args.min_hold, adx_min=args.adx_min, cost=args.cost,
                         mdc_l_min=args.mdc_l_min, mdc_u_max=args.mdc_u_max, mdc_max_width=args.mdc_max_width)
    if mdc_res is None:
        # 売り帯が制約で見つからない場合は、制約を少し緩めてリトライ（幅+10°）
        mdc_res = search_mdc(df, ogc, odc, th, rsi, adx,
                             mgc_fixed=best_mgc, step=args.step, min_trades_mdc=max(3,args.min_trades_mdc-1),
                             min_hold=args.min_hold, adx_min=args.adx_min, cost=args.cost,
                             mdc_l_min=max(0,args.mdc_l_min-5), mdc_u_max=min(90,args.mdc_u_max+5),
                             mdc_max_width=min(90,args.mdc_max_width+10))
    best_mdc = mdc_res["mdc"] if mdc_res is not None else (50,65)  # 最後の砦（妥当域）

    # 最終シミュレーション
    trades, met = simulate(df, ogc, odc, th, rsi, adx,
                           mgc_band=best_mgc, mdc_band=best_mdc,
                           rsi_exit=args.rsi_exit, min_hold=args.min_hold, adx_min=args.adx_min, cost=args.cost)

    # 表示名
    name_map = {}
    name_map.update(JP_NAME_MAP)
    name_map.update(load_name_csv(args.name_map))
    display_name = name_map.get(args.ticker, args.ticker)

    # 保存
    trades_csv  = os.path.join(args.outdir, f"{args.ticker.replace('.','_')}_trades.csv")
    metrics_csv = os.path.join(args.outdir, f"{args.ticker.replace('.','_')}_metrics.csv")
    if trades:
        tdf = pd.DataFrame(trades)
        tdf.insert(0, "ticker", args.ticker)
        tdf.insert(1, "name_jp", display_name)
        tdf["ret_pct"] = (tdf["ret"]*100).round(2)
        tdf["entry_angle"] = tdf["entry_angle"].round(2)
        tdf["exit_angle"]  = tdf["exit_angle"].round(2)
        tdf.to_csv(trades_csv, index=False, encoding="utf-8-sig")
    md = {
        "ticker": args.ticker, "name_jp": display_name,
        "mgc_L": best_mgc[0], "mgc_U": best_mgc[1],
        "mdc_L": best_mdc[0], "mdc_U": best_mdc[1],
        "trades": met["trade_count"], "win_rate": met["win_rate"],
        "total_return": met["total_return"], "cagr": met["cagr"], "max_dd": met["max_dd"],
        "period_start": met["start"], "period_end": met["end"],
        "k": args.k, "step": args.step, "min_hold": args.min_hold,
        "min_trades_mgc": args.min_trades_mgc, "min_trades_mdc": args.min_trades_mdc,
        "adx_min": args.adx_min, "rsi_exit": args.rsi_exit,
        "mdc_l_min": args.mdc_l_min, "mdc_u_max": args.mdc_u_max, "mdc_max_width": args.mdc_max_width,
        "cost": args.cost
    }
    pd.DataFrame([md]).to_csv(metrics_csv, index=False, encoding="utf-8-sig")

    # 出力
    def pct(x): 
        return "nan" if (x is None or (isinstance(x,float) and (np.isnan(x) or np.isinf(x)))) else f"{x*100:.2f}%"
    print(f"===== RESULT ({display_name} / {args.ticker}) =====")
    print(f"MGC (buy angle): {best_mgc}")
    print(f"MDC (sell angle): {best_mdc}")
    print(f"Trades: {met['trade_count']}, WinRate: {pct(met['win_rate'])}")
    print(f"Total Return: {pct(met['total_return'])}, CAGR: {pct(met['cagr'])}")
    print(f"Max Drawdown: {pct(met['max_dd'])}")
    print(f"Period: {met['start']} -> {met['end']}")
    print(f"[SAVE] Trades  -> {trades_csv}")
    print(f"[SAVE] Metrics -> {metrics_csv}")
    print("=======================================")

if __name__ == "__main__":
    main()
