"""Shared engine: mirrors the 6if Structure & Liquidity Sweep Engine (Pine v6).

Used by BOTH the Streamlit dashboard (app.py) and the scheduled alert bot
(alert_bot.py), so the logic only lives in one place.
"""
from __future__ import annotations

import html
import time
from typing import Optional

import numpy as np
import pandas as pd
import requests

# Timeframes used by the matrix / confluence.
TF_ORDER = ["Monthly", "Weekly", "Daily", "H4", "H1", "M15"]
DEFAULT_STRENGTH = {tf: 3 for tf in TF_ORDER}

# Pine's exact tie rule for equal highs/lows in ta.pivothigh/ta.pivotlow is not
# documented. Left side strict, right side non-strict (earliest of equal
# extremes becomes the pivot) is used here. Change in ONE place if you find
# TradingView behaves differently on your symbols.
PIVOT_LEFT_STRICT = True


# --------------------------------------------------------------------------
# Core logic
# --------------------------------------------------------------------------
def raw_pivots(high: np.ndarray, low: np.ndarray, length: int):
    """Pivot values placed at the pivot bar itself (index j).

    A pivot at j can only be *known* at bar j+length (see analyze()).
    """
    n = len(high)
    ph = np.full(n, np.nan)
    pl = np.full(n, np.nan)
    for j in range(length, n - length):
        left_h = high[j - length:j]
        right_h = high[j + 1:j + length + 1]
        left_l = low[j - length:j]
        right_l = low[j + 1:j + length + 1]
        hj, lj = high[j], low[j]
        if PIVOT_LEFT_STRICT:
            ok_h = hj > left_h.max() and hj >= right_h.max()
            ok_l = lj < left_l.min() and lj <= right_l.min()
        else:
            ok_h = hj >= left_h.max() and hj >= right_h.max()
            ok_l = lj <= left_l.min() and lj <= right_l.min()
        if ok_h:
            ph[j] = hj
        if ok_l:
            pl[j] = lj
    return ph, pl


def analyze(
    df: pd.DataFrame,
    length: int = 3,
    sweep_bars: int = 10,
    crt_enabled: bool = True,
) -> dict:
    """Bar-by-bar replay of the Pine logic (f_bias + main engine + CRT).

    Pivot lag is reproduced: at bar i the engine only learns about the pivot
    that formed at bar i-length, exactly like ta.pivothigh(high, len, len).
    """
    n = len(df)
    empty = {
        "bias": 0, "event": "Waiting", "zone": "N/A", "eq": np.nan,
        "dealing_high": np.nan, "dealing_low": np.nan,
        "last_event": "Initializing Structure...",
        "next_target": "Waiting for Event...",
        "lines": [], "crt": [], "pivot_high": np.full(n, np.nan),
        "pivot_low": np.full(n, np.nan),
    }
    if n < length * 2 + 1:
        return empty

    o = df["Open"].to_numpy(float)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    idx = df.index
    ph, pl = raw_pivots(h, l, length)

    hi = lo = np.nan
    hi_t = lo_t = None
    hi_broken = lo_broken = False
    bias, ev = 0, "Waiting"
    last_event = "Initializing Structure..."
    next_target = "Waiting for Event..."
    lines = []  # (x0, x1, y, kind, dashed)  kind: "bull" | "bear"

    for i in range(n):
        j = i - length  # pivot confirmed on this bar
        if j >= 0 and not np.isnan(ph[j]):
            hi, hi_t, hi_broken = ph[j], idx[j], False
        if j >= 0 and not np.isnan(pl[j]):
            lo, lo_t, lo_broken = pl[j], idx[j], False

        if not np.isnan(hi) and not hi_broken and h[i] > hi:
            if c[i] > hi:
                hi_broken, bias, ev = True, 1, "BOS Bullish"
                last_event = "BOS Bullish (Body Close)"
                next_target = "Target: Next External High / FVG"
                lines.append((hi_t, idx[i], hi, "bull", False))
            else:
                ev = "Sweep High"
                last_event = "LIQUIDITY SWEEP HIGH (Wick)"
                next_target = "Target: EQ (50%) -> Range Low"
                lines.append((hi_t, idx[i], hi, "bear", True))

        if not np.isnan(lo) and not lo_broken and l[i] < lo:
            if c[i] < lo:
                lo_broken, bias, ev = True, -1, "BOS Bearish"
                last_event = "BOS Bearish (Body Close)"
                next_target = "Target: Next External Low / FVG"
                lines.append((lo_t, idx[i], lo, "bear", False))
            else:
                ev = "Sweep Low"
                last_event = "LIQUIDITY SWEEP LOW (Wick)"
                next_target = "Target: EQ (50%) -> Range High"
                lines.append((lo_t, idx[i], lo, "bull", True))

    eq = (hi + lo) / 2 if not (np.isnan(hi) or np.isnan(lo)) else np.nan
    zone = "N/A" if np.isnan(eq) else ("PREMIUM" if c[-1] > eq else "DISCOUNT")

    # CRT sweeps: Pine -> bar_index >= last_bar_index - sweepBars
    crt = []
    if crt_enabled:
        start = max(1, n - 1 - sweep_bars)
        for i in range(start, n):
            crh, crl = h[i - 1], l[i - 1]
            bull_bar = c[i] > o[i]
            bear_sweep = c[i] < crh if bull_bar else o[i] < crh
            bull_sweep = o[i] > crl if bull_bar else c[i] > crl
            if h[i] > crh and bear_sweep:
                crt.append((idx[i - 1], idx[i], crh))
            if l[i] < crl and bull_sweep:
                crt.append((idx[i - 1], idx[i], crl))

    return {
        "bias": bias, "event": ev, "zone": zone, "eq": eq,
        "dealing_high": hi, "dealing_low": lo,
        "last_event": last_event, "next_target": next_target,
        "lines": lines, "crt": crt, "pivot_high": ph, "pivot_low": pl,
    }


def crt_allowed(interval: str) -> bool:
    """Pine: timeframe.multiplier >= 60 or not timeframe.isminutes."""
    return interval in ("1h", "4h", "1d", "1wk", "1mo")


# --------------------------------------------------------------------------
# Data (direct downloads per timeframe, like request.security)
# --------------------------------------------------------------------------
_OHLC = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}


def _clean(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    cols = ["Open", "High", "Low", "Close"]
    if not all(col in df.columns for col in cols):
        return None
    df = df[cols].dropna()
    return df if not df.empty else None


def _download(symbol: str, interval: str, period: str, retries: int = 3):
    import yfinance as yf  # lazy so tests / imports don't need it

    last_err = None
    for attempt in range(retries):
        try:
            df = yf.download(
                symbol, period=period, interval=interval,
                progress=False, threads=False, auto_adjust=False,
            )
            df = _clean(df)
            if df is not None:
                return df
        except Exception as exc:  # network / yahoo hiccup
            last_err = exc
        time.sleep(1.5 * (attempt + 1))
    if last_err:
        print(f"[warn] download failed {symbol} {interval}: {last_err}")
    return None


def _to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    return df_1h.resample("4h").agg(_OHLC).dropna()


def fetch_chart(symbol: str, interval: str) -> Optional[pd.DataFrame]:
    """Data for the chart timeframe (1d / 1h / 15m / 4h)."""
    if interval == "1d":
        return _download(symbol, "1d", "3y")
    if interval == "1h":
        return _download(symbol, "1h", "180d")
    if interval == "15m":
        return _download(symbol, "15m", "58d")
    if interval == "4h":
        d = _download(symbol, "1h", "360d")
        return _to_4h(d) if d is not None else None
    return None


def fetch_tf_frames(symbol: str, tfs=TF_ORDER) -> dict:
    """Download each timeframe separately (NOT resampled from the chart)."""
    frames = {}
    if "Monthly" in tfs:
        frames["Monthly"] = _download(symbol, "1mo", "max")
    if "Weekly" in tfs:
        frames["Weekly"] = _download(symbol, "1wk", "15y")
    if "Daily" in tfs:
        frames["Daily"] = _download(symbol, "1d", "5y")
    if "H4" in tfs or "H1" in tfs:
        d1h = _download(symbol, "1h", "720d")
        if "H1" in tfs:
            frames["H1"] = d1h
        if "H4" in tfs:
            frames["H4"] = _to_4h(d1h) if d1h is not None else None
    if "M15" in tfs:
        frames["M15"] = _download(symbol, "15m", "58d")
    return frames


def compute_matrix(frames: dict, strengths: Optional[dict] = None) -> dict:
    """tf -> (bias, event, zone) or None when the data could not be fetched."""
    strengths = {**DEFAULT_STRENGTH, **(strengths or {})}
    out = {}
    for tf, df in frames.items():
        if df is None or len(df) < strengths[tf] * 2 + 1:
            out[tf] = None
            continue
        r = analyze(df, strengths[tf], crt_enabled=False)
        out[tf] = (r["bias"], r["event"], r["zone"])
    return out


# --------------------------------------------------------------------------
# Confluence (ANY-OF rule) + de-duplication
# --------------------------------------------------------------------------
def evaluate_confluence(matrix: dict, tfs=("Weekly", "Daily", "H4")) -> dict:
    """Per timeframe: bullish bias + DISCOUNT, or bearish bias + PREMIUM."""
    bull, bear = [], []
    for tf in tfs:
        row = matrix.get(tf)
        if not row:
            continue
        bias, _ev, zone = row
        if bias == 1 and zone == "DISCOUNT":
            bull.append(tf)
        elif bias == -1 and zone == "PREMIUM":
            bear.append(tf)
    return {"bull": bull, "bear": bear}


def decide_alerts(
    symbol_state: dict,
    matrix: dict,
    tfs=("Weekly", "Daily", "H4"),
    min_match: int = 1,
    cooldown_hours: float = 4.0,
    now: Optional[float] = None,
):
    """Returns (alerts, new_state).

    alerts: list of dicts {direction, timeframes, new}
    Only NEWLY matching (timeframe, direction) pairs trigger an alert, and each
    pair has a cooldown so a zone flickering around EQ can't spam you.
    If a timeframe's data failed to download, its previous state is kept.
    """
    now = time.time() if now is None else now
    prev_matches = set(symbol_state.get("matches", []))
    last_alert = dict(symbol_state.get("last_alert", {}))
    conf = evaluate_confluence(matrix, tfs)

    cur = {f"{tf}:{d}" for d, tl in conf.items() for tf in tl}
    for tf in tfs:  # carry over unknown (failed) timeframes
        if matrix.get(tf) is None:
            cur |= {k for k in prev_matches if k.startswith(f"{tf}:")}

    alerts = []
    for direction in ("bull", "bear"):
        matching = [tf for tf in tfs if f"{tf}:{direction}" in cur]
        if len(matching) < min_match:
            continue
        new = [tf for tf in matching if f"{tf}:{direction}" not in prev_matches]
        fresh = []
        for tf in new:
            key = f"{tf}:{direction}"
            if now - last_alert.get(key, 0) >= cooldown_hours * 3600:
                fresh.append(tf)
        if fresh:
            alerts.append({"direction": direction, "timeframes": matching, "new": fresh})
            for tf in fresh:
                last_alert[f"{tf}:{direction}"] = now

    return alerts, {"matches": sorted(cur), "last_alert": last_alert}


# --------------------------------------------------------------------------
# Telegram
# --------------------------------------------------------------------------
def build_message(symbol: str, alert: dict, matrix: dict, tfs) -> str:
    bull = alert["direction"] == "bull"
    head = "🟢 <b>BULLISH</b> setup" if bull else "🔴 <b>BEARISH</b> setup"
    cond = "Bullish bias + DISCOUNT" if bull else "Bearish bias + PREMIUM"
    lines = [
        f"{head} — <code>{html.escape(symbol)}</code>",
        f"{cond} on: <b>{', '.join(alert['timeframes'])}</b>",
        f"New: {', '.join(alert['new'])}",
        "",
    ]
    for tf in tfs:
        row = matrix.get(tf)
        if not row:
            lines.append(f"{tf}: n/a")
            continue
        b, ev, z = row
        bias = {1: "Bullish", -1: "Bearish"}.get(b, "Neutral")
        lines.append(f"{tf}: {bias} / {z} ({html.escape(ev)})")
    return "\n".join(lines)


def send_telegram(token: str, chat_id: str, text: str, timeout: int = 15):
    """Returns (ok, error_message)."""
    if not token or not chat_id:
        return False, "Missing Telegram token or chat id"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id, "text": text,
        "parse_mode": "HTML", "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        if r.status_code == 200:
            return True, ""
        return False, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as exc:
        return False, str(exc)
