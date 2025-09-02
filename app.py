# Market Context Dashboard (robust v2)
# Free sources: Yahoo Finance RSS + yfinance (^VIX, GC=F, sector ETFs, ^TNX fallback)
# Fixes: normalize MultiIndex columns from yfinance to avoid KeyErrors

import io
import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
import yfinance as yf

st.set_page_config(page_title="Market Context Dashboard", layout="wide")
st.title("📊 Market Context Dashboard")
st.caption("Free data dashboard for PUT credit spreads & Covered Calls — robust to source hiccups.")

HEADERS = {"User-Agent": "Mozilla/5.0"}

# ---------------------------
# Helpers
# ---------------------------
@st.cache_data(ttl=600)
def fetch_yahoo_rss(n=8):
    try:
        url = "https://finance.yahoo.com/rss/topstories"
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, features="xml")
        items = soup.find_all("item")
        return [{
            "title": it.title.text if it.title else "Untitled",
            "link": it.link.text if it.link else "",
            "pubDate": it.pubDate.text if it.pubDate else ""
        } for it in items[:n]]
    except Exception as e:
        return [{"title": f"RSS error: {e}", "link": "", "pubDate": ""}]

def _normalize_close(df, tickers):
    """
    Accepts any yfinance.download() frame (single or multiple tickers, single or MultiIndex columns).
    Returns a DataFrame of Close prices with simple, single-level columns named by ticker (or alias).
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # Case A: MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        # Try ["Close"] on level 0 (common when group_by='ticker')
        if "Close" in df.columns.get_level_values(0):
            close = df["Close"].copy()
        # Try extracting level=1 == 'Close' (common when level 0 is tickers)
        elif "Close" in df.columns.get_level_values(1):
            close = df.xs("Close", axis=1, level=1).copy()
        else:
            # As a last resort try any level containing 'Close'
            levels = [lvl for lvl in range(df.columns.nlevels) if "Close" in df.columns.get_level_values(lvl)]
            if levels:
                close = df.xs("Close", axis=1, level=levels[0]).copy()
            else:
                return pd.DataFrame()
        # Ensure simple column names (tickers only)
        if isinstance(close.columns, pd.MultiIndex):
            close.columns = [c[0] if isinstance(c, tuple) else c for c in close.columns]
        return close

    # Case B: Single-level columns (single ticker): expect 'Close'
    if "Close" in df.columns:
        close = df[["Close"]].copy()
        # Name the column to ticker string for consistency
        if isinstance(tickers, str):
            close.columns = [tickers]
        elif isinstance(tickers, (list, tuple)) and len(tickers) == 1:
            close.columns = [tickers[0]]
        return close

    # Unknown shape
    return pd.DataFrame()

@st.cache_data(ttl=900)
def fetch_fred_10y_csv():
    """Try FRED CSV; raise if columns missing."""
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"
    r = requests.get(url, headers=HEADERS, timeout=10)
    r.raise_for_status()
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
    df = df[df["date"] >= (pd.Timestamp.utcnow() - pd.Timedelta(days=120))]
    return df.set_index("date")[["ten_year_yield"]]

@st.cache_data(ttl=900)
def fetch_10y_yield_series():
    """
    Primary: FRED DGS10
    Fallback: ^TNX via Yahoo Finance (quoted at 10x the yield)
    Returns a DataFrame with a single column 'ten_year_yield' (%)
    """
    # Try FRED first
    try:
        return fetch_fred_10y_csv()
    except Exception:
        pass

    # Fallback: ^TNX from Yahoo
    raw = yf.download("^TNX", period="6mo", interval="1d", auto_adjust=False, threads=True, group_by="ticker")
    close = _normalize_close(raw, "^TNX")
    if close.empty:
        raise ValueError("Unable to fetch ^TNX fallback from Yahoo Finance.")
    out = pd.DataFrame(index=close.index)
    # ^TNX is 10x yield (e.g., 42.7 = 4.27%)
    col = close.columns[0]
    out["ten_year_yield"] = close[col] / 10.0
    out.index = out.index.tz_localize(None)
    return out

@st.cache_data(ttl=600)
def fetch_yf_series(tickers, period="1mo", interval="1d"):
    """Download and normalize Close prices for one or more tickers."""
    raw = yf.download(
        tickers,
        period=period,
        interval=interval,
        auto_adjust=False,
        threads=True,
        group_by="ticker"  # ensures consistent structure we then normalize
    )
    return _normalize_close(raw, tickers)

def pct_change_between_first_last(series):
    s = series.dropna()
    if len(s) >= 2:
        return (s.iloc[-1] / s.iloc[0] - 1.0) * 100.0
    return 0.0

# ---------------------------
# 1) News
# ---------------------------
st.subheader("📰 Latest Market News")
for it in fetch_yahoo_rss(n=8):
    title = it["title"]; link = it["link"]; pd_str = f" — _{it['pubDate']}_" if it["pubDate"] else ""
    st.markdown(f"- [{title}]({link}){pd_str}" if link else f"- {title}{pd_str}")

st.divider()

# ---------------------------
# 2) Rates: 10Y Treasury Yield
# ---------------------------
st.subheader("📉 10-Year Treasury Yield (%)")
try:
    teny = fetch_10y_yield_series().tail(90)
    # Make sure it’s a simple one-column frame
    if isinstance(teny.columns, pd.MultiIndex):
        teny.columns = ["ten_year_yield"]
    st.line_chart(teny.rename(columns={"ten_year_yield": "10Y Yield (%)"}))
    st.metric("Latest 10Y yield", f"{float(teny.iloc[-1, 0]):.2f}%")
except Exception as e:
    st.warning(f"Could not load 10Y yield: {e}")

st.divider()

# ---------------------------
# 3) Gold & VIX
# ---------------------------
st.subheader("📈 Gold & VIX")
alias = {"GC=F": "Gold", "^VIX": "VIX"}
close = fetch_yf_series(list(alias.keys()), period="1mo", interval="1d")
if not close.empty:
    close = close.rename(columns=alias)
    st.line_chart(close)
    for col in close.columns:
        st.metric(f"{col} 1M change", f"{pct_change_between_first_last(close[col]):.2f}%")
else:
    st.warning("Could not load Gold/VIX from Yahoo Finance.")

st.divider()

# ---------------------------
# 4) Sector Heatmap (1D % change)
# ---------------------------
st.subheader("🌐 Sector Heatmap (1D % change)")
sectors = {
    "Tech": "XLK", "Financials": "XLF", "Energy": "XLE",
    "Healthcare": "XLV", "Utilities": "XLU", "Real Estate": "XLRE",
    "Industrials": "XLI", "Materials": "XLB", "Consumer Staples": "XLP",
    "Discretionary": "XLY", "Comm Services": "XLC"
}
changes = {}
for name, tic in sectors.items():
    dfp = fetch_yf_series(tic, period="5d", interval="1d")
    if dfp.empty:
        continue
    s = dfp.iloc[:, 0].dropna().tail(3)  # last few points to survive holidays
    if len(s) >= 2:
        changes[name] = round((s.iloc[-1] / s.iloc[-2] - 1.0) * 100.0, 2)

if changes:
    heatmap_df = pd.DataFrame.from_dict(changes, orient="index", columns=["1D %"])
    st.dataframe(heatmap_df.style.background_gradient(cmap="RdYlGn"))
else:
    st.warning("No sector data available right now.")

st.divider()

# ---------------------------
# 5) Summary Cues for Options
# ---------------------------
st.subheader("📌 Summary Cues for Options Tactics")
st.markdown("""
- **Tariff uncertainty** → Expect headline risk; consider smaller size or wider spreads.
- **Rising yields** → Pressure on high-duration equities; favor conservative strikes.
- **Tech underperforms** → Covered calls can work on lagging mega-caps (watch catalysts).
- **Safe-haven bid (Gold ↑ / VIX ↑)** → Premiums richer; PUT credit spreads attractive **if** supports hold.
- **Event risk (jobs/Fed)** → Tighten DTE or reduce exposure into data prints.
""")
