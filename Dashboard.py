# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import altair as alt
import io
from datetime import datetime
from io import BytesIO

# ---------------------- AUTO-REFRESH ----------------------
try:
    from streamlit_autorefresh import st_autorefresh
    _ = st_autorefresh(interval=5 * 60 * 1000, key="portfolio_autorefresh")  # 5 min
except Exception:
    st.sidebar.warning(
        'Optional package "streamlit-autorefresh" not installed. '
        'Install with: pip install streamlit-autorefresh'
    )

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="Live Portfolio Dashboard", layout="wide")

st.title("📈 Live Profit & Loss Dashboard")

# ---------------------- FILE UPLOAD ----------------------
st.sidebar.header("Upload Portfolio Files")
india_file = st.sidebar.file_uploader("Upload Indian Portfolio CSV", type="csv")
us_file = st.sidebar.file_uploader("Upload US Portfolio CSV", type="csv")

currency_choice = st.sidebar.radio("Display Currency", ["INR", "USD"])

# ---------------------- FX RATE ----------------------
@st.cache_data(ttl=300)
def get_fx_rate():
    try:
        fx = yf.download("USDINR=X", period="1d", interval="1d")
        return fx["Close"].iloc[-1]
    except:
        return 83.0  # fallback
fx_rate = get_fx_rate()

# ---------------------- HELPER FUNCTIONS ----------------------
def load_portfolio(file, is_india=True):
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Ticker": "ticker", "Quantity": "shares", "AvgCost": "avg_cost"})
    if is_india:
        df["ticker"] = df["ticker"].astype(str) + ".NS"
        df["currency"] = "INR"
    else:
        df["currency"] = "USD"
    return df

@st.cache_data(ttl=300)
def fetch_quote(ticker):
    try:
        data = yf.Ticker(ticker)
        info = data.info
        hist = data.history(period="1y")
        return {
            "last_price": info.get("currentPrice", np.nan),
            "trailingPE": info.get("trailingPE", np.nan),
            "forwardPE": info.get("forwardPE", np.nan),
            "eps": info.get("trailingEps", np.nan),
            "dividendYield": info.get("dividendYield", np.nan),
            "beta": info.get("beta", np.nan),
            "hist": hist,
        }
    except Exception:
        return {}

def process_portfolio(df):
    results = []
    for _, row in df.iterrows():
        q = fetch_quote(row["ticker"])
        last_price = q.get("last_price", np.nan)
        cost = row["shares"] * row["avg_cost"]
        mv = row["shares"] * last_price if pd.notna(last_price) else np.nan
        pl = mv - cost if pd.notna(mv) else np.nan
        results.append({
            "Ticker": row["ticker"],
            "Shares": row["shares"],
            "Avg Cost": row["avg_cost"],
            "Cost": cost,
            "Last Price": last_price,
            "Market Value": mv,
            "Unrealized P/L": pl,
            "Unrealized P/L %": (pl / cost * 100) if cost > 0 else np.nan,
            "Trailing PE": q.get("trailingPE", np.nan),
            "Forward PE": q.get("forwardPE", np.nan),
            "EPS": q.get("eps", np.nan),
            "Dividend Yield": q.get("dividendYield", np.nan),
            "Beta": q.get("beta", np.nan),
        })
    return pd.DataFrame(results)

# ---------------------- PROCESS FILES ----------------------
india_df = load_portfolio(india_file, is_india=True) if india_file else None
us_df = load_portfolio(us_file, is_india=False) if us_file else None

india_proc = process_portfolio(india_df) if india_df is not None else None
us_proc = process_portfolio(us_df) if us_df is not None else None

# ---------------------- COMBINED PORTFOLIO ----------------------
combined_proc = None
if india_proc is not None and us_proc is not None:
    combined_proc = pd.concat([india_proc, us_proc], ignore_index=True)
elif india_proc is not None:
    combined_proc = india_proc.copy()
elif us_proc is not None:
    combined_proc = us_proc.copy()

# ---------------------- CURRENCY CONVERSION ----------------------
def convert_currency(df, to="INR"):
    if df is None:
        return None
    df = df.copy()
    if to == "USD":
        df["Cost"] = df.apply(lambda r: r["Cost"]/fx_rate if r["Ticker"].endswith(".NS") else r["Cost"], axis=1)
        df["Market Value"] = df.apply(lambda r: r["Market Value"]/fx_rate if r["Ticker"].endswith(".NS") else r["Market Value"], axis=1)
        df["Unrealized P/L"] = df["Market Value"] - df["Cost"]
    else:  # INR
        df["Cost"] = df.apply(lambda r: r["Cost"] if r["Ticker"].endswith(".NS") else r["Cost"]*fx_rate, axis=1)
        df["Market Value"] = df.apply(lambda r: r["Market Value"] if r["Ticker"].endswith(".NS") else r["Market Value"]*fx_rate, axis=1)
        df["Unrealized P/L"] = df["Market Value"] - df["Cost"]
    df["Unrealized P/L %"] = (df["Unrealized P/L"]/df["Cost"]*100).round(2)
    return df

india_proc = convert_currency(india_proc, to=currency_choice)
us_proc = convert_currency(us_proc, to=currency_choice)
combined_proc = convert_currency(combined_proc, to=currency_choice)

# ---------------------- OVERALL SUMMARY ----------------------
if combined_proc is not None:
    total_cost = combined_proc["Cost"].sum()
    total_mv = combined_proc["Market Value"].sum()
    total_pl = total_mv - total_cost
    total_pl_pct = (total_pl/total_cost*100) if total_cost > 0 else np.nan

    st.subheader("Overall Portfolio Summary")
    st.metric("Total Cost", f"{total_cost:,.2f} {currency_choice}")
    st.metric("Market Value", f"{total_mv:,.2f} {currency_choice}")
    st.metric("Unrealized P/L", f"{total_pl:,.2f} {currency_choice}", f"{total_pl_pct:.2f}%")

# ---------------------- BENCHMARK COMPARISON ----------------------
st.subheader("Portfolio vs Benchmarks")
benchmarks = {"S&P 500": "^GSPC", "NASDAQ": "^IXIC", "NIFTY50": "^NSEI"}

def get_perf(ticker):
    try:
        hist = yf.download(ticker, period="1y", interval="1d")
        return hist["Close"].pct_change().add(1).cumprod()-1
    except:
        return pd.Series(dtype=float)

if combined_proc is not None:
    perf_df = pd.DataFrame()
    perf_df["Portfolio"] = (
        combined_proc["Market Value"].sum()/combined_proc["Cost"].sum()-1
    )
    chart_data = pd.DataFrame()
    for name, symbol in benchmarks.items():
        chart_data[name] = get_perf(symbol)
    chart_data = chart_data.dropna()
    chart_data.index = chart_data.index.date

    chart = alt.Chart(chart_data.reset_index().melt("Date")).mark_line().encode(
        x="Date:T", y=alt.Y("value:Q", axis=alt.Axis(format="%")), color="variable:N"
    ).properties(title="Benchmark Performance (1Y)")
    st.altair_chart(chart, use_container_width=True)

# ---------------------- PORTFOLIO TABS ----------------------
if combined_proc is not None:
    tabs = st.tabs(["Indian Portfolio", "US Portfolio", "Combined Portfolio"])
    portfolios = {"Indian Portfolio": india_proc, "US Portfolio": us_proc, "Combined Portfolio": combined_proc}

    for tab, df in zip(tabs, portfolios.values()):
        with tab:
            if df is not None:
                st.dataframe(df, use_container_width=True)

                # EPS Projection (example chart per ticker)
                st.markdown("**EPS Projection (10 years)**")
                for _, row in df.iterrows():
                    if pd.notna(row["EPS"]) and row["EPS"] > 0:
                        eps_proj = [row["EPS"]*((1+0.08)**i) for i in range(11)]  # assume 8% growth
                        proj_df = pd.DataFrame({"Year": range(11), "EPS": eps_proj})
                        proj_chart = alt.Chart(proj_df).mark_line().encode(x="Year", y="EPS").properties(
                            title=f"{row['Ticker']} EPS Projection"
                        )
                        st.altair_chart(proj_chart, use_container_width=True)

                # Download buttons
                csv = df.to_csv(index=False).encode()
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{tab.lower().replace(' ','_')}.csv",
                    mime="text/csv",
                )

# ---------------------- EXCEL EXPORT ----------------------
if combined_proc is not None:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        if india_proc is not None:
            india_proc.to_excel(writer, sheet_name="India", index=False)
        if us_proc is not None:
            us_proc.to_excel(writer, sheet_name="US", index=False)
        combined_proc.to_excel(writer, sheet_name="Combined", index=False)
    st.sidebar.download_button(
        "Download Excel (All Portfolios)",
        data=output.getvalue(),
        file_name="portfolio_summary.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
