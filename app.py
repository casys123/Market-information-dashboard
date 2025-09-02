# Market Context Dashboard — robust v3
# Free sources only: Yahoo Finance RSS + yfinance (^VIX, GC=F, sector ETFs, ^TNX fallback)
# Fixes:
# - Normalizes yfinance MultiIndex frames
# - FRED DGS10 -> fallback to ^TNX (yield/10)
# - Sector heatmap gracefully degrades if matplotlib not installed
# - Adds "Signals" row to guide PUT credit spreads & Covered Calls

import io
import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
import yfinance as yf

st.set_page_config(page_title="Market Context Dashboard", layout="wide")
st.title("📊 Market Context Dashboard")
st.caption("Free, resilient data for better PUT credit spread & Covered Call decisions.")

HEADERS = {"User-Agent": "Mozilla/5.0"}

# =========================
# Utilities
# =========================
def _has_matplotlib():
    try:
        import matplotlib  # noqa: F401
        return True
    except Exception:
        return False

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
    """Return a DataFrame of Close prices with single-level columns."""
    if df is None or df.empty:
        return pd.DataFrame()

    # MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        # Common forms: level 0 = field, level 1 = ticker OR vice versa
        if "Close" in df.columns.get_level_values(0):
            close = df["Close"].copy()
        elif "Close" in df.columns.get_level_values(1):
            close = df.xs("Close", axis=1, level=1).copy()
        else:
            # last resort: find a level that contains 'Close'
            levels = [lvl for lvl in range(df.columns.nlevels)
                      if "Close" in df.columns.get_level_values(lvl)]
            if not levels:
                return pd.DataFrame()
            close = df.xs("Close", axis=1, level=levels[0]).copy()
        # ensure simple column names
        if isinstance(close.columns, pd.MultiIndex):
            close.columns = [c[0] if isinstance(c, tuple) else c for c in close.columns]
        return close

    # Single-level columns (likely single ticker)
    if "Close" in df.columns:
        close = df[["Close"]].copy()
        # name the column to ticker for consistency
        if isinstance(tickers, str):
            close.columns = [tickers]
        elif isinstance(tickers, (list, tuple)) and len(tickers) == 1:
            close.columns = [tickers[0]]
        return close

    return pd.DataFrame()

@st.cache_data(ttl=900)
def fetch_fred_10y_csv():
    """Try FRED CSV for DGS10; raise if missing columns or parse error."""
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"
    r = requests.get(url, headers=HEADERS, timeout=10)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
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
    """Primary: FRED DGS10. Fallback: ^TNX (divide by 10 to get %)."""
    # Try FRED
    try:
        return fetch_fred_10y_csv()
    except Exception:
        pass
    # Fallback to Yahoo
    raw = yf.download("^TNX", period="6mo", interval="1d", auto_adjust=False, threads=True, group_by="ticker")
    close = _normalize_close(raw, "^TNX")
    if close.empty:
        raise ValueError("Unable to fetch ^TNX fallback from Yahoo Finance.")
    out = pd.DataFrame(index=close.index)
    out["ten_year_yield"] = close.iloc[:, 0] / 10.0  # ^TNX is 10x yield
    out.index = out.index.tz_localize(None)
    return out

@st.cache_data(ttl=600)
def fetch_yf_series(tickers, period="1mo", interval="1d"):
    """Download & normalize Close prices for one or more tickers."""
    raw = yf.download(tickers, period=period, interval=interval, auto_adjust=False, threads=True, group_by="ticker")
    return _normalize_close(raw, tickers)

def pct_change_first_last(series):
    s = series.dropna()
    if len(s) >= 2:
        return (s.iloc[-1] / s.iloc[0] - 1.0) * 100.0
    return 0.0

def pct_change_last_two(series):
    s = series.dropna()
    if len(s) >= 2:
        return (s.iloc[-1] / s.iloc[-2] - 1.0) * 100.0
    return 0.0

# =========================
# 0) Signals row (quick cues)
# =========================
def compute_signals():
    """Returns dict of signals to guide trade bias."""
    signals = []

    # 10Y yield
    try:
        ten = fetch_10y_yield_series().iloc[:, 0].dropna()
        last_10y = float(ten.iloc[-1])
    except Exception:
        last_10y = None

    # Gold & VIX (1M window)
    alias = {"GC=F": "Gold", "^VIX": "VIX"}
    gv = fetch_yf_series(list(alias.keys()), period="1mo", interval="1d")
    if not gv.empty:
        gv = gv.rename(columns=alias)
        gold_1m = pct_change_first_last(gv["Gold"]) if "Gold" in gv.columns else None
        vix_1m  = pct_change_first_last(gv["VIX"])  if "VIX"  in gv.columns else None
        vix_last = gv["VIX"].dropna().iloc[-1] if "VIX" in gv.columns and gv["VIX"].dropna().size else None
    else:
        gold_1m = vix_1m = vix_last = None

    # Sector pulse — Tech (XLK) and Defensives (XLV)
    tech = fetch_yf_series("XLK", period="5d", interval="1d")
    hlth = fetch_yf_series("XLV", period="5d", interval="1d")
    tech_1d = pct_change_last_two(tech.iloc[:, 0]) if not tech.empty else None
    hlth_1d = pct_change_last_two(hlth.iloc[:, 0]) if not hlth.empty else None

    # Build signals with thresholds
    if vix_last is not None:
        signals.append(("VIX", f"{vix_last:.1f}", "↑ Premium rich" if vix_last >= 20 else "Normal"))
    if last_10y is not None:
        signals.append(("10Y Yield", f"{last_10y:.2f}%", "↑ Growth headwind" if last_10y >= 4.25 else "Neutral/Tailwind"))
    if gold_1m is not None:
        signals.append(("Gold 1M", f"{gold_1m:+.2f}%", "↑ Risk-off" if gold_1m >= 2.0 else "Neutral"))
    if tech_1d is not None:
        signals.append(("XLK 1D", f"{tech_1d:+.2f}%", "Tech weak" if tech_1d <= -1.0 else "Stable/Strong"))
    if hlth_1d is not None:
        signals.append(("XLV 1D", f"{hlth_1d:+.2f}%", "Defensive bid" if hlth_1d >= 0.5 else "Neutral"))

    # Simple guidance blurb
    guidance = []
    if vix_last is not None and vix_last >= 20:
        guidance.append("Premiums ↑ → good for **credit** strategies (PUT spreads, covered calls); mind gap risk.")
    if last_10y is not None and last_10y >= 4.25:
        guidance.append("High yields → pressure on growth/AI; pick **conservative strikes** / lower beta where possible.")
    if tech_1d is not None and tech_1d <= -1.0:
        guidance.append("Tech weak → consider **covered calls** on lagging mega-caps (watch catalysts).")
    if gold_1m is not None and gold_1m >= 2.0:
        guidance.append("Risk-off tone → tighten DTE/size; sell puts only near strong supports.")
    if not guidance:
        guidance.append("Neutral backdrop → standard rules; favor liquid tickers with clear support/resistance.")

    return signals, " • ".join(guidance)

# =========================
# Layout
# =========================

# Signals row
st.subheader("⚡ Signals")
sig, blurb = compute_signals()
if sig:
    sig_df = pd.DataFrame(sig, columns=["Indicator", "Value", "Interpretation"])
    st.dataframe(sig_df, hide_index=True, use_container_width=True)
st.info(blurb)

st.divider()

# 1) News
st.subheader("📰 Latest Market News")
for it in fetch_yahoo_rss(n=8):
    title = it["title"]
    link = it["link"]
    pd_str = f" — _{it['pubDate']}_" if it["pubDate"] else ""
    st.markdown(f"- [{title}]({link}){pd_str}" if link else f"- {title}{pd_str}")

st.divider()

# 2) 10Y Treasury Yield
st.subheader("📉 10-Year Treasury Yield (%)")
try:
    teny = fetch_10y_yield_series().tail(90)
    if isinstance(teny.columns, pd.MultiIndex):
        teny.columns = ["ten_year_yield"]
    st.line_chart(teny.rename(columns={"ten_year_yield": "10Y Yield (%)"}))
    st.metric("Latest 10Y yield", f"{float(teny.iloc[-1, 0]):.2f}%")
except Exception as e:
    st.warning(f"Could not load 10Y yield: {e}")

st.divider()

# 3) Gold & VIX
st.subheader("📈 Gold & VIX")
alias = {"GC=F": "Gold", "^VIX": "VIX"}
gv = fetch_yf_series(list(alias.keys()), period="1mo", interval="1d")
if not gv.empty:
    gv = gv.rename(columns=alias)
    st.line_chart(gv)
    for col in gv.columns:
        st.metric(f"{col} 1M change", f"{pct_change_first_last(gv[col]):.2f}%")
else:
    st.warning("Could not load Gold/VIX from Yahoo Finance.")

st.divider()

# 4) Sector Heatmap (1D % change)
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

if not changes:
    st.warning("No sector data available right now.")
else:
    heatmap_df = pd.DataFrame.from_dict(changes, orient="index", columns=["1D %"])
    if _has_matplotlib():
        st.dataframe(heatmap_df.style.background_gradient(cmap="RdYlGn"))
    else:
        # Fallback: simple signal column with emojis (no matplotlib required)
        def colorize(x):
            if x >= 0.5: return "🟢"
            if x <= -0.5: return "🔴"
            return "🟡"
        display_df = heatmap_df.copy()
        display_df["Signal"] = display_df["1D %"].apply(lambda v: f"{colorize(v)} {v:+.2f}%")
        st.dataframe(display_df[["Signal"]], use_container_width=True)

st.divider()

# 5) Option Tactics Cheat Sheet (always-visible)
st.subheader("📌 Tactics Cheat Sheet (quick use)")
st.markdown("""
- **PUT Credit Spreads**: Favor when **VIX elevated (≥20)** but price above strong support; choose **65%+ POP**, **7–21 DTE**,
  sell below support with room to breathe; avoid large event risk (earnings/FOMC/jobs) inside DTE.
- **Covered Calls**: Favor when **IV is rich** or **sector/stock is lagging**; sell slightly OTM strikes, **7–21 DTE**,
  roll if strength returns; avoid short calls through major upside catalysts unless comfortable selling shares.
""")
