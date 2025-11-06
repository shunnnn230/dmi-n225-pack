# filename: n225_batch.py
# -*- coding: utf-8 -*-
import os, re, csv, json, math, datetime as dt, subprocess, sys, hashlib
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import yfinance as yf
from ta.trend import ADXIndicator
from ta.momentum import RSIIndicator

ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "out" / "n225"
CACHE_DIR = OUT_ROOT / "_cache"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PY  = sys.executable if os.environ.get("CI") == "1" else str(ROOT / ".venv" / "Scripts" / "python.exe")
CLI = str(ROOT / "sony_mgc_mdc.py")

PARAMS = {
    "period": "5y", "interval": "1d",
    "dmi": "14", "rsi": "14", "k": "5",
    "step": "5", "min_hold": "8",
    "min_trades_mgc": "8", "min_trades_mdc": "5",
    "adx_min": "15", "rsi_exit": "80", "cost": "0.002",
    "mdc_l_min": "45", "mdc_u_max": "70", "mdc_max_width": "20",
}

NIKKEI_COMPONENTS_EN = "https://indexes.nikkei.co.jp/en/nkave/index/component?idx=nk225"
TFS_COMPONENTS       = "https://topforeignstocks.com/indices/the-components-of-the-nikkei-225-index/"
UA_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept-Language": "ja,en-US;q=0.8,en;q=0.7",
    "Referer": "https://indexes.nikkei.co.jp/en/nkave/",
    "Connection": "keep-alive",
}

def _save_cache(rows):
    p = CACHE_DIR / "constituents.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["ticker","name"])
        for r in rows: w.writerow([r["ticker"], r.get("name_jp") or r.get("name") or ""])
    return p

def _load_cache():
    p = CACHE_DIR / "constituents.csv"
    if not p.exists(): return None
    out = []
    with p.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            t = (r.get("ticker") or "").strip()
            if t.endswith(".T"): out.append({"ticker": t, "name_jp": r.get("name","")})
    return out or None

def fetch_from_nikkei():
    with requests.Session() as s:
        r = s.get(NIKKEI_COMPONENTS_EN, headers=UA_HEADERS, timeout=20)
        r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    rows, seen = [], set()
    for tr in soup.select("table tr"):
        tds = tr.find_all("td")
        if len(tds) < 2: continue
        code = tds[0].get_text(strip=True); name = tds[1].get_text(strip=True)
        if re.fullmatch(r"\d{4}", code) and code not in seen:
            rows.append({"code": code, "ticker": f"{code}.T", "name_jp": name}); seen.add(code)
    return rows if len(rows) >= 200 else []

def fetch_from_topforeignstocks():
    r = requests.get(TFS_COMPONENTS, headers=UA_HEADERS, timeout=20); r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    rows, seen = [], set()
    for td in soup.find_all(["td","p","li"]):
        txt = td.get_text(" ", strip=True)
        m = re.search(r"(\d{4})\.T", txt)
        if m:
            code = m.group(1); name = txt.split(m.group(0))[0].strip(" ,\u3000")
            if re.fullmatch(r"\d{4}", code) and code not in seen:
                rows.append({"code": code, "ticker": f"{code}.T", "name_jp": name}); seen.add(code)
    return rows if len(rows) >= 200 else []

def fetch_from_local_csv():
    nm = ROOT / "name_map.csv"
    if not nm.exists(): return []
    out, seen = [], set()
    with nm.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            t = (r.get("ticker") or "").strip()
            if t.endswith(".T") and t not in seen:
                out.append({"code": t.split(".")[0], "ticker": t, "name_jp": r.get("name_jp","")}); seen.add(t)
    return out

def fetch_n225_codes_and_names():
    cached = _load_cache()
    if cached:
        return [{"code": c["ticker"].split(".")[0], "ticker": c["ticker"], "name_jp": c["name_jp"]} for c in cached]
    try:
        rows = fetch_from_nikkei()
        if rows: _save_cache(rows); return rows
    except Exception: pass
    try:
        rows = fetch_from_topforeignstocks()
        if rows: _save_cache(rows); return rows
    except Exception: pass
    rows = fetch_from_local_csv()
    if rows: _save_cache(rows); return rows
    raise RuntimeError("N225 構成銘柄の取得に失敗。name_map.csv を用意してください。")

def run_one(ticker: str):
    args = [
        PY, CLI,
        "--ticker", ticker,
        "--period", PARAMS["period"], "--interval", PARAMS["interval"],
        "--dmi", PARAMS["dmi"], "--rsi", PARAMS["rsi"], "--k", PARAMS["k"],
        "--step", PARAMS["step"], "--min-hold", PARAMS["min_hold"],
        "--min-trades-mgc", PARAMS["min_trades_mgc"], "--min-trades-mdc", PARAMS["min_trades_mdc"],
        "--adx-min", PARAMS["adx_min"], "--rsi-exit", PARAMS["rsi_exit"], "--cost", PARAMS["cost"],
        "--mdc-l-min", PARAMS["mdc_l_min"], "--mdc-u-max", PARAMS["mdc_u_max"], "--mdc-max-width", PARAMS["mdc_max_width"],
        "--outdir", str(ROOT / "out")
    ]
    sp = subprocess.run(args, capture_output=True, text=True)
    ok = (sp.returncode == 0)
    pfx = ticker.replace(".", "_")
    metrics_csv = ROOT / "out" / f"{pfx}_metrics.csv"
    return ok and metrics_csv.exists(), metrics_csv

def _angle_from_spread(S: pd.Series, k: int = 5) -> pd.Series:
    th = np.full(len(S), np.nan); x = np.arange(k)
    for i in range(k-1, len(S)):
        w = S.iloc[i-k+1:i+1].values
        if np.isnan(w).any(): continue
        m = np.polyfit(x, w, 1)[0]
        th[i] = np.degrees(np.arctan(m))
    return pd.Series(th, index=S.index, name="theta")

def _compute_signal_today(ticker: str, mgc_band, mdc_band,
                          dmi_win=14, rsi_win=14, k=5, adx_min=15) -> dict:
    df = yf.download(ticker, period="180d", interval="1d", auto_adjust=True, progress=False, threads=False)
    if df is None or len(df) < (k+2):
        return {"signal":"HOLD","reason":"NO_DATA","angle":math.nan,"adx":math.nan,"rsi":math.nan}
    adx = ADXIndicator(df["High"], df["Low"], df["Close"], window=dmi_win, fillna=False)
    di_p, di_n, adx_v = adx.adx_pos(), adx.adx_neg(), adx.adx()
    rsi = RSIIndicator(df["Close"], window=rsi_win, fillna=False).rsi()
    S   = di_p - di_n
    th  = _angle_from_spread(S, k=k).abs()
    i = len(df)-1
    prev, cur = S.iat[i-1], S.iat[i]
    is_gc = (prev <= 0) and (cur > 0)
    is_dc = (prev >= 0) and (cur < 0)
    ang   = th.iat[i] if not np.isnan(th.iat[i]) else math.nan
    adx_t = adx_v.iat[i] if not np.isnan(adx_v.iat[i]) else math.nan
    rsi_t = rsi.iat[i] if not np.isnan(rsi.iat[i]) else math.nan
    if not math.isnan(rsi_t) and rsi_t >= 80:
        return {"signal":"SELL","reason":"RSI","angle":float(ang), "adx":float(adx_t), "rsi":float(rsi_t)}
    if is_dc and (mdc_band is not None) and (not math.isnan(ang)) and (mdc_band[0] <= ang <= mdc_band[1]):
        return {"signal":"SELL","reason":"DC","angle":float(ang), "adx":float(adx_t), "rsi":float(rsi_t)}
    if is_gc and (mgc_band is not None) and (not math.isnan(ang)) and (mgc_band[0] <= ang <= mgc_band[1]) and (not math.isnan(adx_t)) and adx_t >= adx_min:
        return {"signal":"BUY","reason":"GC","angle":float(ang), "adx":float(adx_t), "rsi":float(rsi_t)}
    return {"signal":"HOLD","reason":"NONE","angle":float(ang) if not math.isnan(ang) else math.nan,
            "adx":float(adx_t) if not math.isnan(adx_t) else math.nan,
            "rsi":float(rsi_t) if not math.isnan(rsi_t) else math.nan}

def _sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    rows = fetch_n225_codes_and_names()
    jst = dt.datetime.now(dt.timezone(dt.timedelta(hours=9)))
    ymd = jst.strftime("%Y%m%d")
    day_out = OUT_ROOT / ymd
    day_out.mkdir(parents=True, exist_ok=True)

    merged = []
    for i, r in enumerate(rows, 1):
        tkr = r["ticker"]; nm = r.get("name_jp","")
        print(f"[{i}/{len(rows)}] {tkr} {nm} ...")
        ok, mpath = run_one(tkr)
        if ok:
            with mpath.open("r", encoding="utf-8") as f:
                row = next(csv.DictReader(f), None)
            if row:
                merged.append({
                    "ticker": tkr, "name_jp": nm,
                    "mgc": f"{row['mgc_L']}-{row['mgc_U']}",
                    "mdc": f"{row['mdc_L']}-{row['mdc_U']}",
                    "trades": row["trades"], "win_rate": row["win_rate"],
                    "total_return": row["total_return"], "cagr": row["cagr"], "max_dd": row["max_dd"],
                    "start": row.get("period_start") or row.get("start",""),
                    "end":   row.get("period_end")   or row.get("end",""),
                    "updated_at": jst.isoformat(timespec="seconds"),
                })
            (day_out / mpath.name).write_bytes(mpath.read_bytes())
        else:
            merged.append({
                "ticker": tkr, "name_jp": nm, "mgc":"", "mdc":"", "trades":"0",
                "win_rate":"", "total_return":"", "cagr":"", "max_dd":"",
                "start":"", "end":"", "updated_at": jst.isoformat(timespec="seconds"),
            })

    metrics_path = day_out / "metrics_all.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        cols = ["ticker","name_jp","mgc","mdc","trades","win_rate","total_return","cagr","max_dd","start","end","updated_at"]
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in merged: w.writerow(r)
    (OUT_ROOT / "latest_metrics.csv").write_bytes(metrics_path.read_bytes())
    print(f"[OK] Saved: {metrics_path}")
    print(f"[OK] Saved: {OUT_ROOT / 'latest_metrics.csv'}")

    mdf = pd.read_csv(OUT_ROOT / "latest_metrics.csv")
    sig_rows = []
    for _, r in mdf.iterrows():
        tkr = r["ticker"]
        try:
            mgc = str(r["mgc"]).split("-"); mdc = str(r["mdc"]).split("-")
            mgc_band = (int(float(mgc[0])), int(float(mgc[1]))) if len(mgc)==2 else None
            mdc_band = (int(float(mdc[0])), int(float(mdc[1]))) if len(mdc)==2 else None
            info = _compute_signal_today(tkr, mgc_band, mdc_band,
                                         dmi_win=int(PARAMS["dmi"]), rsi_win=int(PARAMS["rsi"]),
                                         k=int(PARAMS["k"]), adx_min=float(PARAMS["adx_min"]))
            sig_rows.append({
                "ticker": tkr, "name_jp": r.get("name_jp",""),
                "signal": info["signal"], "reason": info["reason"],
                "angle_today": info.get("angle", math.nan),
                "adx_today": info.get("adx", math.nan),
                "rsi_today": info.get("rsi", math.nan),
                "updated_at": jst.isoformat(timespec="seconds"),
            })
        except Exception:
            sig_rows.append({
                "ticker": tkr, "name_jp": r.get("name_jp",""),
                "signal": "HOLD","reason":"ERROR",
                "angle_today": math.nan, "adx_today": math.nan, "rsi_today": math.nan,
                "updated_at": jst.isoformat(timespec="seconds"),
            })
    sig_df = pd.DataFrame(sig_rows)
    sig_path = day_out / "signals_today.csv"
    sig_df.to_csv(sig_path, index=False, encoding="utf-8-sig")
    (OUT_ROOT / "latest_signals.csv").write_bytes(sig_path.read_bytes())
    print(f"[OK] Saved: {sig_path}")
    print(f"[OK] Saved: {OUT_ROOT / 'latest_signals.csv'}")

    m = mdf.copy()
    m["total_return"] = pd.to_numeric(m["total_return"], errors="coerce")
    top = m.sort_values("total_return", ascending=False).head(10)
    top_cols = ["ticker","name_jp","mgc","mdc","trades","win_rate","total_return","cagr","max_dd"]
    top_rows = top[top_cols].to_dict(orient="records")

    q = m["total_return"].quantile([0.2,0.4,0.6,0.8]).values.tolist() if m["total_return"].notna().any() else [0,0,0,0]
    def to_star(x):
        if pd.isna(x): return 1
        if x <= q[0]: return 1
        if x <= q[1]: return 2
        if x <= q[2]: return 3
        if x <= q[3]: return 4
        return 5
    stars = m[["ticker","name_jp","total_return"]].copy()
    stars["stars"] = stars["total_return"].apply(to_star)
    sig_latest = pd.read_csv(OUT_ROOT / "latest_signals.csv") if (OUT_ROOT / "latest_signals.csv").exists() else pd.DataFrame()
    metrics_cols = ["ticker","name_jp","mgc","mdc","trades","win_rate","total_return","cagr","max_dd","start","end","updated_at"]
    pack = {
        "updated_at": dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(timespec="seconds"),
        "top": top_rows,
        "stars": stars[["ticker","name_jp","total_return","stars"]].to_dict(orient="records"),
        "signals": sig_latest.to_dict(orient="records") if not sig_latest.empty else [],
        "metrics": m[metrics_cols].to_dict(orient="records")
    }
    pack_path_day = day_out / "pack.json"
    with pack_path_day.open("w", encoding="utf-8") as f: json.dump(pack, f, ensure_ascii=False)
    (OUT_ROOT / "latest_pack.json").write_bytes(pack_path_day.read_bytes())
    print(f"[OK] Saved: {pack_path_day}")
    print(f"[OK] Saved: {OUT_ROOT / 'latest_pack.json'}")

    def _sha256_of(p: Path) -> str:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1<<20), b""):
                h.update(chunk)
        return h.hexdigest()

    ver = {
        "updated_at": pack["updated_at"],
        "sha256": _sha256_of(OUT_ROOT / "latest_pack.json"),
        "size": (OUT_ROOT / "latest_pack.json").stat().st_size
    }
    with (OUT_ROOT / "version.json").open("w", encoding="utf-8") as f:
        json.dump(ver, f, ensure_ascii=False)
    print(f".[OK] Saved: {OUT_ROOT / 'version.json'}")

if __name__ == "__main__":
    main()
