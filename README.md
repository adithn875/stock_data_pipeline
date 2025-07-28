
# 📈 Stock Data Analysis Pipeline

This project is a complete **Stock Data Analysis Pipeline** that fetches, processes, analyzes, and visualizes stock market data. It provides an interactive **Streamlit dashboard** for exploring insights, viewing tables, downloading data, and summarizing stock trends.

---

## 🚀 Features

- 📦 Modular Python-based ETL pipeline
- 📊 Visual analytics using Plotly and Matplotlib
- 🧮 Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
- 📉 Interactive charts with Streamlit
- 🧾 Export processed data to Excel
- 🔒 Logging for traceability
- 📂 Clean project structure with reusable components

---

## 🧱 Project Structure

```bash
├── main.py                  # Entry-point CLI
├── fetch_data.py            # Fetch stock data from yFinance
├── process_data.py          # Clean & process fetched data
├── visualize_data.py        # Plotting & technical indicators
├── dashboard.py             # Streamlit UI
├── utils/
│   └── logger.py            # Logging utility
├── screenshots/
│   └── dashboard/           # Images of the dashboard interface
│       ├── interface.png
│       ├── summary.png
│       ├── analysis.png
│       ├── datatable.png
│       └── download_data.png
```

---

## 🖼️ Streamlit Dashboard Snapshots

| Interface | Summary | Technical Analysis |
|-----------|---------|--------------------|
| ![](screenshots/dashboard/interface.png) | ![](screenshots/dashboard/summary.png) | ![](screenshots/dashboard/analysis.png) |

| Data Table | Download Feature |
|------------|------------------|
| ![](screenshots/dashboard/datatable.png) | ![](screenshots/dashboard/download_data.png) |

---

## 📦 Setup Instructions

1. **Clone the Repository**

```bash
git clone https://github.com/your-username/stock-data-pipeline.git
cd stock-data-pipeline
```

2. **Create Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
```

3. **Install Requirements**

```bash
pip install -r requirements.txt
```

4. **Run Streamlit Dashboard**

```bash
streamlit run dashboard.py
```

---

## 📍 Use Cases

- Track and visualize live stock trends
- Apply technical indicators to support investment decisions
- Export stock insights to Excel for reports

---

## 🤝 Contributions

Feel free to fork, modify, or raise pull requests to enhance this project.

---

## 📧 Contact

For feedback or collaboration:  
📬 adithnk07@gmail.com

---

**Built with ❤️ by Adith Nk**
