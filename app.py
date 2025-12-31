import numpy as np
import pandas as pd
import streamlit as st
from polygon import RESTClient
import ta

# ======================================================
# CONFIG
# ======================================================
st.set_page_config(layout="wide")
st.title("🏛️ Scanner Institutionnel SWING — S&P 500 (Edge Score)")

client = RESTClient(st.secrets["POLYGON_API_KEY"])
LOOKBACK = 220

# ======================================================
# LOAD SP500 + SECTORS
# ======================================================
@st.cache_data
def load_sp500():
    df = pd.read_excel("sp500_constituents.xlsx")
    df["Symbol"] = df["Symbol"].astype(str).str.replace(".", "-", regex=False)
    return df[["Symbol", "Sector"]]

sp500 = load_sp500()
TICKERS = sp500["Symbol"].tolist()

st.caption(f"📦 {len(TICKERS)} tickers S&P 500 chargés")

# ======================================================
# DATA FETCH
# ======================================================
@st.cache_data
def get_data(ticker, mult, span):
    bars = client.get_aggs(ticker, mult, span, limit=LOOKBACK)
    df = pd.DataFrame([{
        "close": b.close,
        "high": b.high,
        "low": b.low
    } for b in bars])
    return df

# ======================================================
# EDGE SCORE
# ======================================================
def edge_score(df):
    ema = ta.trend.ema_indicator(df["close"], 50)
    atr = ta.volatility.average_true_range(df["high"], df["low"], df["close"])
    rsi = ta.momentum.rsi(df["close"], 14)
    macd_hist = ta.trend.macd_diff(df["close"])

    z1 = (df["close"] - ema) / atr
    z2 = (rsi - 50) / 50
    z3 = macd_hist / atr

    return np.tanh(z1 + z2) * np.tanh(z3)

# ======================================================
# MARKET REGIME (SPY)
# ======================================================
spy_df = get_data("SPY", 1, "day")
edge_spy = edge_score(spy_df).iloc[-1]

# ======================================================
# VIX FILTER
# ======================================================
vix_df = get_data("I:VIX", 1, "day")
vix_z = (vix_df["close"].iloc[-1] - vix_df["close"].mean()) / vix_df["close"].std()
vix_filter = 1 - np.tanh(vix_z)

# ======================================================
# BREADTH OFFICIEL SP500
# ======================================================
@st.cache_data
def compute_sp500_breadth(tickers):
    hits, valid = 0, 0
    for t in tickers:
        try:
            df = get_data(t, 1, "day")
            ema = ta.trend.ema_indicator(df["close"], 50)
            if df["close"].iloc[-1] > ema.iloc[-1]:
                hits += 1
            valid += 1
        except:
            continue
    return hits / valid if valid > 0 else 0

BREADTH = compute_sp500_breadth(TICKERS)

# ======================================================
# REGIME FLAGS
# ======================================================
LONG_OK  = edge_spy > 0.3 and vix_filter > 0.3 and BREADTH > 0.5
SHORT_OK = edge_spy < -0.3 and vix_filter > 0.3 and BREADTH < 0.5

# ======================================================
# SCAN 500 STOCKS (SWING)
# ======================================================
rows = []

for t in TICKERS:
    try:
        d1 = get_data(t, 1, "day")
        h4 = get_data(t, 4, "hour")

        e1d = edge_score(d1).iloc[-1]
        e4h = edge_score(h4).iloc[-1]

        rows.append({
            "Ticker": t,
            "Edge 1D": e1d,
            "Edge 4H": e4h
        })
    except:
        continue

scan_df = pd.DataFrame(rows)

# ======================================================
# TOP 10 LONG / SHORT
# ======================================================
top_long = scan_df[
    (LONG_OK) &
    (scan_df["Edge 1D"] > 0.3) &
    (scan_df["Edge 4H"] > 0.4)
].sort_values("Edge 4H", ascending=False).head(10)

top_short = scan_df[
    (SHORT_OK) &
    (scan_df["Edge 1D"] < -0.3) &
    (scan_df["Edge 4H"] < -0.4)
].sort_values("Edge 4H").head(10)

# ======================================================
# SECTOR HEATMAP (MEAN EDGE 4H)
# ======================================================
sector_scores = {}

for sector in sp500["Sector"].unique():
    tickers = sp500[sp500["Sector"] == sector]["Symbol"]
    edges = []

    for t in tickers:
        try:
            h4 = get_data(t, 4, "hour")
            edges.append(edge_score(h4).iloc[-1])
        except:
            continue

    if len(edges) > 0:
        sector_scores[sector] = np.mean(edges)

sector_df = pd.DataFrame.from_dict(
    sector_scores, orient="index", columns=["Mean Edge 4H"]
).sort_values("Mean Edge 4H", ascending=False)

# ======================================================
# DISPLAY
# ======================================================
st.subheader("🌍 Régime Marché")
c1, c2, c3 = st.columns(3)
c1.metric("SPY Edge", round(edge_spy, 2))
c2.metric("VIX Filter", round(vix_filter, 2))
c3.metric("Breadth S&P 500", f"{BREADTH*100:.1f}%")

st.subheader("🟢 TOP 10 LONG — SWING")
st.dataframe(top_long, use_container_width=True)

st.subheader("🔴 TOP 10 SHORT — SWING")
st.dataframe(top_short, use_container_width=True)

st.subheader("🧭 Heatmap Sectorielle — Mean Edge 4H")
st.dataframe(sector_df, use_container_width=True)
