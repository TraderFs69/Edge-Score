import numpy as np
import pandas as pd
import streamlit as st
from polygon import RESTClient
import ta
from datetime import datetime, timedelta

# ======================================================
# CONFIG
# ======================================================
st.set_page_config(layout="wide")
st.title("🏛️ Scanner Institutionnel SWING — S&P 500 (Edge Score)")

client = RESTClient(st.secrets["POLYGON_API_KEY"])
LOOKBACK = 220

USER_MODE = st.radio(
    "Mode de sélection",
    ["STRICT", "RELATIF"],
    horizontal=True
)

# ======================================================
# LOAD SP500
# ======================================================
@st.cache_data
def load_sp500():
    df = pd.read_excel("sp500_constituents.xlsx")
    df["Symbol"] = df["Symbol"].astype(str).str.replace(".", "-", regex=False)
    return df[["Symbol", "Sector"]]

sp500 = load_sp500()
TICKERS = sp500["Symbol"].tolist()

st.caption(f"📦 Univers : {len(TICKERS)} actions (S&P 500)")

# ======================================================
# POLYGON DATA (SAFE + CACHE)
# ======================================================
@st.cache_data
def get_data(ticker, mult, span):
    to_date = datetime.utcnow()
    from_date = to_date - timedelta(days=LOOKBACK * 3)

    bars = client.get_aggs(
        ticker=ticker,
        multiplier=mult,
        timespan=span,
        from_=from_date.strftime("%Y-%m-%d"),
        to=to_date.strftime("%Y-%m-%d"),
        limit=LOOKBACK
    )

    return pd.DataFrame(
        [{"close": b.close, "high": b.high, "low": b.low} for b in bars]
    )

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
# PRELOAD DATA (ASSOUPLI)
# ======================================================
data_1d, data_4h = {}, {}

with st.spinner("Chargement des données (premier lancement plus lent)…"):
    for t in TICKERS:
        try:
            d1 = get_data(t, 1, "day")
            h4 = get_data(t, 4, "hour")

            if len(d1) >= 40 and len(h4) >= 20:
                data_1d[t] = d1
                data_4h[t] = h4
        except:
            continue

# ======================================================
# MARKET SCORES (AFFICHÉS)
# ======================================================
spy_df = get_data("SPY", 1, "day")
edge_spy = edge_score(spy_df).iloc[-1]

vix_df = get_data("VIXY", 1, "day")
vix_z = (vix_df["close"].iloc[-1] - vix_df["close"].mean()) / vix_df["close"].std()
vix_filter = 1 - np.tanh(vix_z)

hits = 0
for df in data_1d.values():
    ema = ta.trend.ema_indicator(df["close"], 50)
    if df["close"].iloc[-1] > ema.iloc[-1]:
        hits += 1

BREADTH = hits / len(data_1d) if data_1d else 0

LONG_OK  = edge_spy > 0.3 and vix_filter > 0.3 and BREADTH > 0.5
SHORT_OK = edge_spy < -0.3 and vix_filter > 0.3 and BREADTH < 0.5

# ======================================================
# SCAN
# ======================================================
rows = []

for t in data_1d:
    try:
        e1d = edge_score(data_1d[t]).iloc[-1]
        e4h = edge_score(data_4h[t]).iloc[-1]

        rows.append({
            "Ticker": t,
            "Edge 1D": e1d,
            "Edge 4H": e4h
        })
    except:
        continue

scan_df = pd.DataFrame(rows)

# ======================================================
# BULLETPROOF RANKING
# ======================================================
if "Edge 4H" in scan_df.columns and not scan_df.empty:
    scan_df["Rank_4H"] = scan_df["Edge 4H"].rank(pct=True)
else:
    scan_df["Rank_4H"] = np.nan

# ======================================================
# MODE EFFECTIF
# ======================================================
MODE = USER_MODE
if scan_df["Rank_4H"].notna().sum() < 10:
    st.warning("Peu de données exploitables → passage automatique en mode RELATIF")
    MODE = "RELATIF"

# ======================================================
# TOP 10
# ======================================================
if MODE == "STRICT":
    top_long = scan_df[
        LONG_OK &
        (scan_df["Edge 1D"] > 0.3) &
        (scan_df["Edge 4H"] > 0.25)
    ].sort_values("Edge 4H", ascending=False).head(10)

    top_short = scan_df[
        SHORT_OK &
        (scan_df["Edge 1D"] < -0.3) &
        (scan_df["Edge 4H"] < -0.25)
    ].sort_values("Edge 4H").head(10)
else:
    top_long = scan_df.sort_values("Rank_4H", ascending=False).head(10)
    top_short = scan_df.sort_values("Rank_4H").head(10)

# ======================================================
# SECTOR HEATMAP
# ======================================================
sector_scores = {}

for sector in sp500["Sector"].unique():
    edges = []

    for t in sp500[sp500["Sector"] == sector]["Symbol"]:
        h4 = data_4h.get(t)
        if h4 is None:
            continue

        e = edge_score(h4).iloc[-1]
        if not np.isnan(e):
            edges.append(e)

    if len(edges) >= 5:
        sector_scores[sector] = np.mean(edges)

sector_df = pd.DataFrame.from_dict(
    sector_scores, orient="index", columns=["Mean Edge 4H"]
)

if not sector_df.empty:
    sector_df["Rank"] = sector_df["Mean Edge 4H"].rank(pct=True)

# ======================================================
# DISPLAY — SCORES COMME TANTÔT
# ======================================================
st.subheader("🌍 Scores de Marché")
c1, c2, c3, c4 = st.columns(4)
c1.metric("SPY Edge", round(edge_spy, 2))
c2.metric("VIX Filter", round(vix_filter, 2))
c3.metric("Breadth (1D)", f"{BREADTH*100:.1f}%")
c4.metric("Mode Actif", MODE)

st.caption(
    f"🔍 {len(scan_df)} actions analysées | "
    f"LONG_OK={LONG_OK} | SHORT_OK={SHORT_OK}"
)

st.subheader(f"🟢 TOP 10 LONG — {MODE}")
st.dataframe(top_long, use_container_width=True)

st.subheader(f"🔴 TOP 10 SHORT — {MODE}")
st.dataframe(top_short, use_container_width=True)

st.subheader("🧭 Heatmap Sectorielle — Mean Edge 4H")
st.dataframe(sector_df.sort_values("Mean Edge 4H", ascending=False), use_container_width=True)
