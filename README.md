# 🏆 Vietnam Gold Price Pipeline & Forecast Dashboard

This is a complete end-to-end data pipeline project. Its primary goal is to **automatically collect**, **store**, **process**, and **visualize** gold price data from the Vietnamese market (focusing on SJC, PNJ, and DOJI).

The system is fully automated and integrates Machine Learning models to analyze and forecast future price trends.

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://gold-pipeline-nhom-5.streamlit.app/)

> **Note:** The dashboard is hosted on Streamlit Cloud's free tier and may take a moment to wake up from sleep.
> * **Live Dashboard URL:** `https://nhom5vuive.streamlit.app/`
> * **GitHub Repo URL:** `https://github.com/duyenhai1015/goldd`

---

## ⚙️ Pipeline Architecture

The system is 100% cloud-based and fully automated. It operates on a "Warehouse - Staff - Storefront" model:



### 1. 📦 The Warehouse (Database): MongoDB Atlas
* **Role:** The central, 24/7 data warehouse that stores all historical and real-time data.
* **Details:** A free-tier cluster on MongoDB Atlas is used for secure and scalable storage.
* **Security:** Access is strictly limited via IP Whitelisting and Environment Variables (Secrets).

### 2. 🚚 The Staff (Automation): GitHub Actions
The system has two automated "workers" (workflows) running 24/7:

* **Workflow 1: Daily Backfill (`backfill.yml`)**
    * **Schedule:** Runs once per day (at 00:00 UTC / 7:00 AM Vietnam Time).
    * **Task:** Executes `backfill_data.py` (a heavy script using `pandas`) to generate historical data for PNJ, SJC, and DOJI for the *previous day* and load it into Atlas.
    * **Anti-Duplication:** A unique index on the database ensures that duplicate records are automatically and safely ignored.

* **Workflow 2: Real-time Scraper (`scraper.yml`)**
    * **Schedule:** Runs every 60 minutes.
    * **Task:** Executes the lightweight `scraper.py` script to scrape the latest DOJI prices and `upsert` (update or insert) them into Atlas.

### 3. 🛍️ The Storefront (Dashboard): Streamlit Cloud
* **Role:** The front-end web interface for the end-user.
* **File:** `dashboard.py`
* **Logic:**
    1.  Securely connects to MongoDB Atlas via the `MONGODB_ATLAS_URI` Secret.
    2.  Fetches all data from the warehouse.
    3.  Uses `@st.cache_data(ttl=60)` (a 60-second cache) to ensure high performance while keeping data fresh.
    4.  Renders interactive charts and runs ML models.

---

## 📊 Dashboard Features

The `dashboard.py` (v5.11) interface provides the following features:

* **Multi-Brand Theming:** The dashboard's logo, color scheme, and theme automatically change based on the selected brand (PNJ, SJC, or DOJI).
* **Dynamic Filtering:** Allows users to filter all charts and metrics by:
    * Brand (Defaults to DOJI).
    * Gold Type (e.g., Rings, SJC Bars, Jewelry...).
    * Date Range (Start Date, End Date).
* **Visual Analysis:** Four main tabs for analysis (Buy Price, Sell Price, Spread) and one tab for detailed row-level data.
* **Machine Learning Forecast Center:**
    1.  Automatically compares the performance (MAE score) of 3 models: **Linear Regression**, **Random Forest**, and **XGBoost**.
    2.  Selects the best-performing model for the filtered data.
    3.  Re-trains the best model on the complete dataset.
    4.  Plots a 30-day future forecast against the historical data.

---

## 🛠️ Tech Stack & Libraries

| Category | Technology |
| :--- | :--- |
| **Automation** | GitHub Actions |
| **Database** | MongoDB Atlas (Cloud NoSQL) |
| **Dashboard** | Streamlit (Streamlit Cloud) |
| **Data Scraping** | `requests`, `BeautifulSoup4` |
| **Data Handling** | `pandas`, `numpy` |
| **Visualization** | `plotly` (integrated with Streamlit) |
| **Machine Learning**| `scikit-learn`, `xgboost` |

*(**Note on PySpark:** This project uses **Pandas** instead of PySpark. This was a deliberate technical decision, as the dataset size (~10k-100k rows) is "Small Data," which Pandas handles with high efficiency. PySpark requires a complex, costly cluster and is unnecessary (overkill) for this project's scale and deployment environment.)*

---

## 🏗️ Directory Structure
