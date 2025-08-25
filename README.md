# Live P&L Dashboard (Streamlit)

This repository contains a Streamlit app that provides a **live profit & loss dashboard** for managing Indian (NSE) and US (NYSE/NASDAQ) stock portfolios. The app follows value investing principles inspired by Warren Buffett, Charlie Munger, and Peter Lynch.

---

## 🚀 Features
- Upload **two CSVs**: one for Indian stocks (NSE, tickers with `.NS` suffix added automatically) and one for US stocks.
- **Overall Summary** (in USD or INR): Total cost, market value, unrealized P/L, and portfolio % return.
- **Benchmarks**: Compare portfolio performance against S&P 500, NASDAQ, and NIFTY50.
- **Tabbed Views**:
  - **Indian Portfolio**: Holdings, valuation ratios, EPS projections.
  - **US Portfolio**: Holdings, valuation ratios, EPS projections.
  - **Combined Portfolio**: Unified view across both regions.
- **Stock health indicators**: Trailing PE, Forward PE, EPS, Dividend Yield, Beta, 12M Avg PE.
- **10-year EPS Projections**: Based on trailing EPS and growth estimates.
- **Download Options**:
  - CSV per tab.
  - Excel export (with 3 sheets: Indian, US, Combined).
- **Auto-refresh**: Updates every 5 minutes (via `streamlit-autorefresh`).

---

## 📂 File Structure
