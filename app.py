import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ict_engine as eng
import providers as prov

st.set_page_config(
    page_title="ICT Structure & Liquidity Sweep Engine [v6]",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ 6if Structure & Liquidity Sweep Engine [v6]")
st.markdown(
    "Mirrors the TradingView Pine v6 indicator: major pivots (with Pine's"
    " confirmation lag), BOS / sweeps, CRT window, EQ line and a multi-timeframe"
    " matrix built from real per-timeframe data."
)

# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Configuration & Tickers")
ticker_options = [
    "deriv:frxEURUSD", "deriv:frxGBPUSD", "deriv:frxXAUUSD", "deriv:frxXAGUSD",
    "deriv:R_100", "deriv:BOOM1000", "deriv:CRASH1000",
    "ccxt:kraken:BTC/USDT", "ccxt:kraken:ETH/USDT", "Custom...",
]
choice = st.sidebar.selectbox("Select Asset", ticker_options)
if choice == "Custom...":
    symbol = st.sidebar.text_input(
        "Custom spec", value="deriv:frxUSDJPY",
        help="deriv:<code> | ccxt:<exchange>:<PAIR> | yf:<yahoo ticker>",
    ).strip()
else:
    symbol = choice

with st.sidebar.expander("🔎 Find Deriv symbol codes"):
    st.caption("Search, then copy the spec into Custom... or config.json")

    @st.cache_data(ttl=3600, show_spinner="Asking Deriv for symbols...")
    def _deriv_list():
        return prov.deriv_symbols()

    q = st.text_input("Search (e.g. gold, volatility, boom, eur)", key="dsym_q")
    try:
        rows = _deriv_list()
        words = q.lower().split()
        shown = [
            r for r in rows
            if all(w in f"{r['spec']} {r['name']} {r['market']} {r['submarket']}".lower() for w in words)
        ]
        st.caption(f"{len(shown)} of {len(rows)} symbols")
        st.dataframe(
            pd.DataFrame(shown)[["spec", "name", "market"]] if shown else pd.DataFrame(),
            hide_index=True, use_container_width=True, height=300,
        )
    except Exception as exc:
        st.error(f"Could not load Deriv symbols: {exc}")

interval = st.sidebar.selectbox("Chart Timeframe", ["1d", "1h", "15m", "4h"], index=0)

st.sidebar.subheader("Pivot Settings")
major_len = st.sidebar.number_input("Major Swing Length", 2, 50, 3, 1)
sweep_bars = st.sidebar.number_input("CRT Sweep Look Bar Window", 1, 50, 10, 1)

with st.sidebar.expander("Per-timeframe strengths (matrix)"):
    strengths = {
        tf: st.number_input(f"{tf} Strength", 1, 50, 3, 1, key=f"str_{tf}")
        for tf in eng.TF_ORDER
    }

st.sidebar.subheader("🔔 Confluence rule")
conf_tfs = st.sidebar.multiselect(
    "Timeframes to check", eng.TF_ORDER, default=["Weekly", "Daily", "H4"]
)
min_match = st.sidebar.number_input(
    "Minimum timeframes matching", 1, max(1, len(conf_tfs)), 1, 1,
    help="1 = ANY timeframe (bias + zone agree). 3 = all three must agree.",
)

st.sidebar.subheader("📲 Telegram (manual button only)")


def _secret(name: str) -> str:
    try:
        return st.secrets.get(name, "") or os.environ.get(name, "")
    except Exception:  # no secrets.toml present
        return os.environ.get(name, "")


token = _secret("TELEGRAM_BOT_TOKEN")
chat_id = _secret("TELEGRAM_CHAT_ID")
if not token:
    token = st.sidebar.text_input("Telegram Bot Token", type="password")
if not chat_id:
    chat_id = st.sidebar.text_input("Telegram Chat ID")
st.sidebar.caption(
    "Automatic alerts run via GitHub Actions (alert_bot.py), not from this page."
)

st.sidebar.subheader("Chart Visual Toggles")
show_dots = st.sidebar.checkbox("Show Major Pivot Dots (last 100 bars)", True)
show_lines = st.sidebar.checkbox("Show BOS / Sweep Lines", True)
show_eq = st.sidebar.checkbox("Show Equilibrium (EQ) Line", True)

st.sidebar.subheader("Color Settings")
bull_color = st.sidebar.color_picker("Bullish / Discount Color", "#00FF00")
bear_color = st.sidebar.color_picker("Bearish / Premium Color", "#FF0000")
eq_color = st.sidebar.color_picker("Equilibrium (EQ) Color", "#FFFF00")


# ---------------- Data ----------------
@st.cache_data(ttl=60, show_spinner="Loading chart data...")
def load_chart(sym, itv):
    return prov.fetch_chart(sym, itv)


@st.cache_data(ttl=60, show_spinner="Loading multi-timeframe data...")
def load_matrix(sym, strength_items):
    frames = prov.fetch_tf_frames(sym)
    return eng.compute_matrix(frames, dict(strength_items))


df = load_chart(symbol, interval)
if df is None or df.empty:
    st.error("No data for this asset/timeframe. Check the spec, e.g. deriv:frxEURUSD or ccxt:kraken:BTC/USDT.")
    st.stop()

# ---------------- Main-chart analysis ----------------
res = eng.analyze(
    df, major_len, sweep_bars, crt_enabled=eng.crt_allowed(interval)
)
matrix = load_matrix(symbol, tuple(sorted(strengths.items())))

dealing_high, dealing_low, eq = res["dealing_high"], res["dealing_low"], res["eq"]
active_zone = res["zone"]


def fmt(x):
    return "N/A" if x is None or np.isnan(x) else f"{x:.2f}"


# ---------------- Dashboard ----------------
st.subheader(f"📊 ICT Structure Engine Dashboard ({symbol}, {interval})")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Dealing High", fmt(dealing_high))
    st.metric("Pricing Zone", active_zone)
with c2:
    st.metric("Equilibrium (50% EQ)", fmt(eq))
    st.metric("Last Structure Event", res["last_event"])
with c3:
    st.metric("Dealing Low", fmt(dealing_low))
    st.metric("Dynamic Target Objective", res["next_target"])

st.markdown("### 🌐 Multi-Timeframe Macro Alignment Matrix")
cols = st.columns(len(eng.TF_ORDER))
for col, tf in zip(cols, eng.TF_ORDER):
    row = matrix.get(tf)
    with col:
        if not row:
            st.markdown(f"**{tf}**\n\nNo data")
            continue
        b, ev, z = row
        bias_badge = "🟢 Bullish" if b == 1 else ("🔴 Bearish" if b == -1 else "⚪ Neutral")
        zone_badge = "🔴 PREMIUM" if z == "PREMIUM" else ("🟢 DISCOUNT" if z == "DISCOUNT" else "⚪ N/A")
        st.markdown(f"**{tf}**\n\nEvent: `{ev}`\n\nBias: {bias_badge}\n\nZone: **{zone_badge}**")

# ---------------- Confluence ----------------
st.markdown("---")
st.subheader("🔔 Confluence (any-of rule)")
conf = eng.evaluate_confluence(matrix, tuple(conf_tfs))
st.write(
    f"**Bullish (bias Bullish + DISCOUNT):** {', '.join(conf['bull']) or '—'}  \n"
    f"**Bearish (bias Bearish + PREMIUM):** {', '.join(conf['bear']) or '—'}"
)

if st.button("🚀 Send current confluence to Telegram"):
    sent_any = False
    for direction in ("bull", "bear"):
        tl = conf[direction]
        if len(tl) >= min_match and tl:
            alert = {"direction": direction, "timeframes": tl, "new": tl}
            text = eng.build_message(symbol, alert, matrix, tuple(conf_tfs))
            ok, err = eng.send_telegram(token, chat_id, text)
            if ok:
                sent_any = True
                st.success(f"{direction.upper()} alert sent.")
            else:
                st.error(f"Telegram error: {err}")
    if not sent_any and not any(len(conf[d]) >= min_match and conf[d] for d in conf):
        st.info("No confluence match right now, nothing sent.")

st.markdown("---")

# ---------------- Chart ----------------
st.subheader("📈 Interactive Price & Structure Chart")
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price",
))

if show_dots:
    n = len(df)
    mask = np.arange(n) >= n - 100
    ph = pd.Series(res["pivot_high"], index=df.index).where(mask).dropna()
    pl = pd.Series(res["pivot_low"], index=df.index).where(mask).dropna()
    fig.add_trace(go.Scatter(x=ph.index, y=ph.values, mode="markers",
                             marker=dict(size=11, color=bear_color), name="Major Swing High"))
    fig.add_trace(go.Scatter(x=pl.index, y=pl.values, mode="markers",
                             marker=dict(size=11, color=bull_color), name="Major Swing Low"))

if show_eq and not np.isnan(eq):
    step = df.index.to_series().diff().median()
    x0 = df.index[max(0, len(df) - 1 - 15)]
    x1 = df.index[-1] + 5 * step
    fig.add_shape(type="line", x0=x0, x1=x1, y0=eq, y1=eq,
                  line=dict(color=eq_color, width=1.5, dash="dot"))
    fig.add_annotation(x=x1, y=eq, text="EQ (50%)", showarrow=False,
                       font=dict(color=eq_color), xanchor="left")

if show_lines:
    for x0, x1, y, kind, dashed in res["lines"][-500:]:
        fig.add_shape(type="line", x0=x0, x1=x1, y0=y, y1=y,
                      line=dict(color=bull_color if kind == "bull" else bear_color,
                                width=1.5, dash="dash" if dashed else "solid"))
    for x0, x1, y in res["crt"]:
        fig.add_shape(type="line", x0=x0, x1=x1, y0=y, y1=y,
                      line=dict(color="#4B4B4B", width=1))

fig.update_layout(
    title=f"{symbol} - Structure & Liquidity Engine",
    xaxis_title="Timeline", yaxis_title="Price",
    template="plotly_dark", height=650, xaxis_rangeslider_visible=False,
)
st.plotly_chart(fig, use_container_width=True)
