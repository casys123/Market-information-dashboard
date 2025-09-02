# app.py
import requests
import pandas as pd
import yfinance as yf
import streamlit as st
from bs4 import BeautifulSoup
import datetime

st.set_page_config(page_title="Market Context Dashboard", layout="wide")
st.title("📊 Market Context Dashboard")
st.write("Free data dashboard to support option strategies (PUT credit spreads & Covered Calls).")

# ---------------------------
# 1. News Headlines (Yahoo RSS)
# ---------------------------
st.header("📰 Latest Market News")
url = "https://finance.yahoo.com/rss/topstories"
resp = requests.get(url)
soup = BeautifulSoup(resp.content, features="xml")
items = soup.findAll("item")[:5]
for item in items:
    st.markdown(f"- [{item.title.text}]({item.link.text})")

# ---------------------------
# 2. Treasury Yields (10Y from FRED)
# ---------------------------
st.header("📉 Treasury Yield (10Y)")
fred_series = "DGS10"
fred_api = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=" + fred_series
df = pd.read_csv(fred_api, parse_dates=["DATE"])
df = df.dropna().tail(60)
st.line_chart(df.set_index("DATE"))

# ---------------------------
# 3. Gold & VIX (Yahoo Finance)
# ---------------------------
st.header("📈 Gold & VIX")
tickers = {"Gold": "GC=F", "VIX": "^VIX"}
data = yf.download(list(tickers.values()), period="5d", interval="1d")["Close"]

st.line_chart(data.rename(columns={v: k for k,v in tickers.items()}))

# ---------------------------
# 4. Sector Heatmap (SPDR ETFs)
# ---------------------------
st.header("🌐 Sector Heatmap")
sectors = {
    "Tech": "XLK", "Financials": "XLF", "Energy": "XLE",
    "Healthcare": "XLV", "Utilities": "XLU", "Real Estate": "XLRE",
    "Industrials": "XLI", "Materials": "XLB", "Consumer Staples": "XLP",
    "Discretionary": "XLY", "Comm Services": "XLC"
}
prices = {}
for sec, ticker in sectors.items():
    df = yf.download(ticker, period="5d", interval="1d")["Close"]
    pct_change = (df.iloc[-1] / df.iloc[0] - 1) * 100
    prices[sec] = pct_change.round(2)

heatmap_df = pd.DataFrame(prices, index=["5D % Change"]).T
st.dataframe(heatmap_df.style.background_gradient(cmap="RdYlGn"))

# ---------------------------
# 5. Summary Digest
# ---------------------------
st.header("📌 Summary Digest")
st.markdown("""
- **Tariffs**: Trade-policy uncertainty still overhangs.
- **Bonds**: Rising yields (~4.2–4.3%) pressure equities.
- **Tech/AI**: NVDA, AAPL, MSFT underperforming.
- **Safe-havens**: Gold up, VIX elevated.
- **Macro caution**: Awaiting jobs data, post-holiday volatility.
""")
