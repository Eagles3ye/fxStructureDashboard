"""Data providers.

Symbol specs (used in config.json and the app):
    deriv:frxEURUSD        forex / metals / synthetics via Deriv WebSocket API
    ccxt:kraken:BTC/USDT   crypto via ccxt (exchange is part of the spec)
    yf:AAPL  or  AAPL      Yahoo fallback (delayed, no synthetics)

Run `python list_deriv_symbols.py` to see the exact Deriv symbol codes.
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

import pandas as pd

import ict_engine as eng

_OHLC = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}

DERIV_GRAN = {"15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
CCXT_TF = {"15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}


def parse_spec(spec: str):
    """-> ("deriv", symbol) | ("ccxt", exchange, symbol) | ("yf", symbol)"""
    spec = spec.strip()
    if spec.startswith("deriv:"):
        return ("deriv", spec[6:])
    if spec.startswith("ccxt:"):
        parts = spec.split(":", 2)
        if len(parts) != 3 or not parts[1] or not parts[2]:
            raise ValueError(f"Bad ccxt spec '{spec}', use ccxt:<exchange>:<SYMBOL>")
        return ("ccxt", parts[1], parts[2])
    if spec.startswith("yf:"):
        return ("yf", spec[3:])
    return ("yf", spec)


# --------------------------------------------------------------------------
# Small TTL cache (matters in --loop mode, where we poll every minute)
# --------------------------------------------------------------------------
_cache: dict = {}
TTL = {"1d": 120, "4h": 55, "1h": 55, "15m": 55}


def _cached(key, ttl, fn):
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    val = fn()
    _cache[key] = (now, val)
    return val


# --------------------------------------------------------------------------
# Resampling helpers (weekly / monthly are built from daily)
# --------------------------------------------------------------------------
def to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    return daily.resample("W-MON", closed="left", label="left").agg(_OHLC).dropna()


def to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    return daily.resample("MS").agg(_OHLC).dropna()


# --------------------------------------------------------------------------
# Deriv
# --------------------------------------------------------------------------
class DerivClient:
    """Minimal synchronous Deriv WebSocket client (one connection, req_id matched)."""

    def __init__(self, app_id: Optional[str] = None, timeout: int = 20):
        self.app_id = app_id or os.environ.get("DERIV_APP_ID", "1089")
        self.timeout = timeout
        self.ws = None
        self._req = 0
        self._last = 0.0

    def _connect(self):
        import websocket  # pip install websocket-client

        self.ws = websocket.create_connection(
            f"wss://ws.derivws.com/websockets/v3?app_id={self.app_id}",
            timeout=self.timeout,
        )

    def close(self):
        try:
            if self.ws:
                self.ws.close()
        finally:
            self.ws = None

    def request(self, payload: dict, _retry: bool = True) -> dict:
        try:
            if self.ws is None:
                self._connect()
            gap = 0.25 - (time.time() - self._last)  # gentle spacing
            if gap > 0:
                time.sleep(gap)
            self._req += 1
            self.ws.send(json.dumps({**payload, "req_id": self._req}))
            while True:
                msg = json.loads(self.ws.recv())
                if msg.get("req_id") == self._req:
                    self._last = time.time()
                    break
        except Exception:
            self.close()
            if _retry:
                time.sleep(1)
                return self.request(payload, _retry=False)
            raise
        err = msg.get("error")
        if err and err.get("code") == "RateLimit" and _retry:
            time.sleep(5)
            return self.request(payload, _retry=False)
        return msg

    def candles(self, symbol: str, granularity: int, count: int) -> pd.DataFrame:
        msg = self.request({
            "ticks_history": symbol, "adjust_start_time": 1, "count": count,
            "end": "latest", "style": "candles", "granularity": granularity,
        })
        if "error" in msg:
            raise RuntimeError(f"Deriv {symbol}: {msg['error'].get('message')}")
        rows = msg.get("candles") or []
        if not rows:
            raise RuntimeError(f"Deriv {symbol}: no candles returned")
        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df["epoch"], unit="s", utc=True)
        df = df.rename(columns=str.capitalize)[["Open", "High", "Low", "Close"]]
        return df.astype(float).dropna()


_deriv: Optional[DerivClient] = None


def get_deriv() -> DerivClient:
    global _deriv
    if _deriv is None:
        _deriv = DerivClient()
    return _deriv


def _deriv_base(symbol: str, itv: str, count: int) -> pd.DataFrame:
    return _cached(
        ("deriv", symbol, itv), TTL[itv],
        lambda: get_deriv().candles(symbol, DERIV_GRAN[itv], count),
    )


def deriv_frames(symbol: str, tfs) -> dict:
    out = {}
    daily = None
    if any(t in tfs for t in ("Daily", "Weekly", "Monthly")):
        daily = _safe(lambda: _deriv_base(symbol, "1d", 2500))
    if "Daily" in tfs:
        out["Daily"] = daily
    if "Weekly" in tfs:
        out["Weekly"] = to_weekly(daily) if daily is not None else None
    if "Monthly" in tfs:
        out["Monthly"] = to_monthly(daily) if daily is not None else None
    if "H4" in tfs:
        out["H4"] = _safe(lambda: _deriv_base(symbol, "4h", 2000))
    if "H1" in tfs:
        out["H1"] = _safe(lambda: _deriv_base(symbol, "1h", 3000))
    if "M15" in tfs:
        out["M15"] = _safe(lambda: _deriv_base(symbol, "15m", 3000))
    return out


# --------------------------------------------------------------------------
# ccxt
# --------------------------------------------------------------------------
_exchanges: dict = {}


def _get_exchange(exchange_id: str):
    if exchange_id not in _exchanges:
        import ccxt

        _exchanges[exchange_id] = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    return _exchanges[exchange_id]


def ccxt_ohlcv(exchange_id: str, symbol: str, timeframe: str, total: int) -> pd.DataFrame:
    """Paginated fetch, newest `total` candles (exchanges cap one call at 300-1000)."""
    ex = _get_exchange(exchange_id)
    tf_ms = int(ex.parse_timeframe(timeframe) * 1000)
    now_ms = int(time.time() * 1000)
    since = now_ms - total * tf_ms
    rows: dict = {}
    for _ in range(25):
        batch = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not batch:
            break
        new = [r for r in batch if r[0] not in rows]
        if not new:
            break
        for r in new:
            rows[r[0]] = r
        last = max(rows)
        if last >= now_ms - tf_ms:
            break
        since = last + tf_ms
    if not rows:
        raise RuntimeError(f"ccxt {exchange_id} {symbol} {timeframe}: no data")
    df = pd.DataFrame(sorted(rows.values()),
                      columns=["ts", "Open", "High", "Low", "Close", "Vol"])
    df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df[["Open", "High", "Low", "Close"]].astype(float).dropna()


def _ccxt_base(ex_id: str, symbol: str, itv: str, total: int) -> pd.DataFrame:
    return _cached(
        ("ccxt", ex_id, symbol, itv), TTL[itv],
        lambda: ccxt_ohlcv(ex_id, symbol, CCXT_TF[itv], total),
    )


def ccxt_frames(ex_id: str, symbol: str, tfs) -> dict:
    out = {}
    daily = None
    if any(t in tfs for t in ("Daily", "Weekly", "Monthly")):
        daily = _safe(lambda: _ccxt_base(ex_id, symbol, "1d", 1500))
    if "Daily" in tfs:
        out["Daily"] = daily
    if "Weekly" in tfs:
        out["Weekly"] = to_weekly(daily) if daily is not None else None
    if "Monthly" in tfs:
        out["Monthly"] = to_monthly(daily) if daily is not None else None
    if "H4" in tfs:
        out["H4"] = _safe(lambda: _ccxt_base(ex_id, symbol, "4h", 1500))
    if "H1" in tfs:
        out["H1"] = _safe(lambda: _ccxt_base(ex_id, symbol, "1h", 2000))
    if "M15" in tfs:
        out["M15"] = _safe(lambda: _ccxt_base(ex_id, symbol, "15m", 2000))
    return out


# --------------------------------------------------------------------------
# Public API used by app.py and alert_bot.py
# --------------------------------------------------------------------------
def _safe(fn):
    try:
        return fn()
    except Exception as exc:
        print(f"[warn] fetch failed: {exc}")
        return None


def fetch_tf_frames(spec: str, tfs=eng.TF_ORDER) -> dict:
    kind, *rest = parse_spec(spec)
    if kind == "deriv":
        return deriv_frames(rest[0], tfs)
    if kind == "ccxt":
        return ccxt_frames(rest[0], rest[1], tfs)
    return eng.fetch_tf_frames(rest[0], tfs)


def fetch_chart(spec: str, interval: str) -> Optional[pd.DataFrame]:
    kind, *rest = parse_spec(spec)
    if kind == "deriv":
        return _safe(lambda: _deriv_base(rest[0], interval, 1500))
    if kind == "ccxt":
        return _safe(lambda: _ccxt_base(rest[0], rest[1], interval, 1500))
    return eng.fetch_chart(rest[0], interval)


def deriv_symbols() -> list:
    """All tradable Deriv symbols: [{symbol, name, market, submarket}]."""
    client = DerivClient()
    try:
        msg = client.request({"active_symbols": "brief", "product_type": "basic"})
    finally:
        client.close()
    if "error" in msg:
        raise RuntimeError(msg["error"].get("message", "Deriv error"))
    return [
        {"spec": f"deriv:{r['symbol']}", "name": r.get("display_name"),
         "market": r.get("market"), "submarket": r.get("submarket")}
        for r in sorted(msg.get("active_symbols", []),
                        key=lambda x: (x.get("market", ""), x.get("symbol", "")))
    ]
