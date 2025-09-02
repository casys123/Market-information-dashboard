# Market Context Dashboard (robust)
# Free sources only: Yahoo Finance RSS + yfinance (^VIX, GC=F, sector ETFs, ^TNX fallback)
# Fixes: FRED CSV fallback + safe parsing + caching

import io
import time
import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(page_title="Market Context Dashboard", layout="wide")
st.title("📊 Market Context Dashboard")
st.caption("Free data dashboard for PUT credit spreads & Covered Calls — robust to source hiccups.")

# ---------------------------
# Helpers & Caching
# ---------------------------
HEADERS = {"User-Agent": "Mozilla/5.0"}

@st.cache_data(ttl=600)
def fetch_yahoo_rss(n=8):
    try:
        url = "https://finance.yahoo.com/rss/topstories"
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, features="xml")
        items = soup.find_all("item")
        out = []
        for it in items[:n]:
            out.append({
                "title": it.title.text if it.title else "Untitled",
                "link": it.link.text if it.link else "",
                "pubDate": it.pubDate.text if it.pubDate else ""
            })
        return out
    except Exception as e:
        return [{"title": f"RSS error: {e}", "link": "", "pubDate": ""}]

@st.cache_data(ttl=900)
def fetch_fred_10y_csv():
    """Try FRED CSV. If invalid/HTML or missing DATE column, raise."""
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"
    r = requests.get(url, headers=HEADERS, timeout=10)
    r.raise_for_status()
    # Sometimes comes back as HTML; guard by trying to read as CSV
    try:
        df = pd.read_csv(io.StringIO(r.text))
    except Exception as e:
        raise ValueError(f"FRED CSV parse error: {e}")
    if "DATE" not in df.columns or "DGS10" not in df.columns:
        raise ValueError("FRED CSV missing expected columns")
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"]).rename(columns={"DATE": "date", "DGS10": "ten_year_yield"})
    df["ten_year_yield"] = pd.to_numeric(df["ten_year_yield"], errors="coerce")
    df = df.dropna(subset=["ten_year_yield"])
    return df

@st.cache_data(ttl=900)
def fetch_10y_yield_series():
    """
    Primary: FRED CSV (DGS10)
    Fallback: ^TNX from Yahoo (note: ^TNX is 10Y yield * 10)
    Returns last ~60 points with datetime index and a 'ten_year_yield' column in %
    """
    try:
        df = fetch_fred_10y_csv()
        df = df[df["date"] >= (pd.Timestamp.utcnow() - pd.Timedelta(days=120))]  # recent window
        return df.set_index("date")[["ten_year_yield"]]
    except Exception:
        # Fallback: ^TNX (CBOE 10-Year Treasury Note Yield Index)
        tnx = yf.download("^TNX", period="6mo", interval="1d", auto_adjust=False, threads=True)
        if tnx.empty:
            raise ValueError("Unable to fetch ^TNX fallback from Yahoo Finance.")
        out = tnx[["Close"]].copy()
        # ^TNX is quoted at 10x yield (e.g., 42.7 = 4.27%)
        out.rename(columns={"Close": "ten_year_yield"}, inplace=True)
        out["ten_year_yield"] = out["ten_year_yield"] / 10.0
        out.index = out.index.tz_localize(None)
        return out

@st.cache_data(ttl=600)
def fetch_yf_series(tickers, period="5d", interval="1d"):
    """Safe wrapper around yfinance.download that always returns a DataFrame of Close prices."""
    try:
        data = yf.download(tickers, period=period, interval=interval, auto_adjust=False, threads=True)
        if isinstance(tickers, list) and len(tickers) > 1:
            close = data["Close"]
        else:
            # single ticker: yfinance returns a Series; make it a DF
            close = data["Close"].to_frame(name=tickers if isinstance(tickers, str) else tickers[0])
        return close
    except Exception as e:
        return pd.DataFrame()

def pct_change_recent(series_or_df):
    """Compute % change between the last two valid points; fallback to first/last."""
    x = series_or_df.dropna()
    if x.shape[0] >= 2:
        return (x.iloc[-1] / x.iloc[-2] - 1.0) * 100.0
    elif x.shape[0] >= 1:
        return pd.Series([0.0], index=x.columns) if hasattr(x, "columns") else 0.0
    else:
        return pd.Series(dtype=float) if hasattr(series_or_df, "columns") else float("nan")

# ---------------------------
# 1) News
# ---------------------------
st.subheader("📰 Latest Market News")
news_items = fetch_yahoo_rss(n=8)
for it in news_items:
    title = it["title"]
    link = it["link"]
    pd_str = f" — _{it['pubDate']}_" if it["pubDate"] else ""
    if link:
        st.markdown(f"- [{title}]({link}){pd_str}")
    else:
        st.markdown(f"- {title}{pd_str}")

st.divider()

# ---------------------------
# 2) Rates: 10Y Treasury Yield
# ---------------------------
st.subheader("📉 10-Year Treasury Yield (%)")
try:
    teny = fetch_10y_yield_series().tail(90)
    st.line_chart(teny)
    last_yield = float(teny["ten_year_yield"].iloc[-1])
    st.metric("Latest 10Y yield", f"{last_yield:.2f}%")
except Exception as e:
    st.warning(f"Could not load 10Y yield: {e}")

st.divider()

# ---------------------------
# 3) Gold & VIX
# ---------------------------
st.subheader("📈 Gold & VIX")
tickers = {"Gold": "GC=F", "VIX": "^VIX"}
close = fetch_yf_series(list(tickers.values()), period="1mo", interval="1d")
if not close.empty:
    close = close.rename(columns={v: k for k, v in tickers.items()})
    st.line_chart(close)
    # Metrics
    for col in close.columns:
        try:
            pct = (close[col].dropna().iloc[-1] / close[col].dropna().iloc[0] - 1) * 100
            st.metric(f"{col} 1M change", f"{pct:.2f}%")
        except Exception:
            pass
else:
    st.warning("Could not load Gold/VIX from Yahoo Finance.")

st.divider()

# ---------------------------
# 4) Sector Heatmap (SPDR ETFs)
# ---------------------------
st.subheader("🌐 Sector Heatmap (1D % change)")
sectors = {
    "Tech": "XLK", "Financials": "XLF", "Energy": "XLE",
    "Healthcare": "XLV", "Utilities": "XLU", "Real Estate": "XLRE",
    "Industrials": "XLI", "Materials": "XLB", "Consumer Staples": "XLP",
    "Discretionary": "XLY", "Comm Services": "XLC"
}

changes = {}
for sec, tic in sectors.items():
    dfp = fetch_yf_series(tic, period="5d", interval="1d")
    if dfp.empty:
        continue
    dfp = dfp.iloc[-3:]  # last few points to survive holidays
    try:
        # 1D change: last vs previous
        pct = pct_change_recent(dfp.squeeze())
        pct_val = float(pct) if not hasattr(pct, "values") else float(pct.values[0])
        changes[sec] = round(pct_val, 2)
    except Exception:
        continue

if changes:
    heatmap_df = pd.DataFrame.from_dict(changes, orient="index", columns=["1D %"])
    st.dataframe(heatmap_df.style.background_gradient(cmap="RdYlGn"))
else:
    st.warning("No sector data available right now.")

st.divider()

# ---------------------------
# 5) Summary Digest (macro cues for options)
# ---------------------------
st.subheader("📌 Summary Cues for Options Tactics")
st.markdown("""
- **Tariff uncertainty** → Expect headline risk; consider smaller size or wider spreads.
- **Rising yields** → Pressure on high-duration equities (mega-cap/AI); favor conservative strikes.
- **Tech underperforms** → Screen for covered calls on weaker mega-caps (higher IV), but mind catalysts.
- **Safe-haven bid (Gold ↑ / VIX ↑)** → Premiums richer; PUT credit spreads can be attractive **if** supports hold.
- **Event risk (jobs/Fed)** → Tighten DTE or reduce exposure into data prints.
""")
