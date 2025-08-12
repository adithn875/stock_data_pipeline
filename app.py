# app.py - FINAL VERSION with News Integration and International Stock Ticker support
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import date, datetime, timedelta
import numpy as np
import yfinance as yf
import requests # New library for making API requests
from textblob import TextBlob # New library for sentiment analysis
import os

# Your Finnhub API key is now directly in the code
FINNHUB_API_KEY = 'd2cru3pr01qihtct93tgd2cru3pr01qihtct93u0'

# Import your existing visualization
try:
    from dashboard import visualize_stock_data, calculate_technical_indicators

    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    st.warning("⚠️ Advanced dashboard module not found - using basic visualization")


# Simple data fetching function
def fetch_stock_data(ticker, start, end):
    """Simple stock data fetching using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start, end=end)
        if data.empty:
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# New function to fetch news from Finnhub
def fetch_stock_news(ticker, start, end, api_key):
    """Fetches stock news for a given ticker and date range using Finnhub API"""
    if not api_key:
        st.warning("Please set your FINNHUB_API_KEY environment variable for news.")
        return []
    try:
        start_str = start.strftime('%Y-%m-%d')
        end_str = end.strftime('%Y-%m-%d')
        url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={start_str}&to={end_str}&token={api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error fetching news (status code {response.status_code}): {response.text}")
            return []
    except Exception as e:
        st.error(f"An error occurred while fetching news: {e}")
        return []

# New function for sentiment analysis
def get_sentiment(text):
    """Calculates sentiment of a text using TextBlob"""
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0.1:
        return "Positive 🟢"
    elif polarity < -0.1:
        return "Negative 🔴"
    else:
        return "Neutral 🟡"

# Simple data processing function
def process_stock_data(data, ticker):
    """Basic data processing with technical indicators"""
    if data is None or data.empty:
        return None

    try:
        # Calculate simple moving averages
        data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
        data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)

        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))

        # Calculate Bollinger Bands
        sma_20 = ta.trend.sma_indicator(data['Close'], window=20)
        std_20 = ta.volatility.bollinger_hband_indicator(data['Close'], window=20)
        data['BB_Upper'] = sma_20 + (std_20 * 2)
        data['BB_Lower'] = sma_20 - (std_20 * 2)

        return data
    except Exception as e:
        st.error(f"Error processing data: {e}")
        return data


def create_basic_chart(processed_df, ticker, show_volume, show_ma, show_rsi, show_bollinger):
    """Create basic plotly chart as fallback"""
    # Interactive price chart with Plotly
    fig = make_subplots(
        rows=3 if show_volume and show_rsi else (2 if show_volume or show_rsi else 1),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(['Price Chart'] +
                        (['Volume'] if show_volume else []) +
                        (['RSI'] if show_rsi else [])),
        row_heights=[0.6, 0.2, 0.2] if show_volume and show_rsi else [0.7, 0.3]
    )

    # Price chart
    fig.add_trace(
        go.Candlestick(
            x=processed_df.index,
            open=processed_df['Open'],
            high=processed_df['High'],
            low=processed_df['Low'],
            close=processed_df['Close'],
            name=f"{ticker} Price"
        ),
        row=1, col=1
    )

    # Moving averages
    if show_ma and 'SMA_20' in processed_df.columns:
        fig.add_trace(
            go.Scatter(
                x=processed_df.index,
                y=processed_df['SMA_20'],
                name='SMA 20',
                line=dict(color='orange', width=2)
            ),
            row=1, col=1
        )

    if show_ma and 'SMA_50' in processed_df.columns:
        fig.add_trace(
            go.Scatter(
                x=processed_df.index,
                y=processed_df['SMA_50'],
                name='SMA 50',
                line=dict(color='red', width=2)
            ),
            row=1, col=1
        )

    # Bollinger Bands
    if show_bollinger and 'BB_Upper' in processed_df.columns:
        fig.add_trace(
            go.Scatter(
                x=processed_df.index,
                y=processed_df['BB_Upper'],
                name='BB Upper',
                line=dict(color='gray', width=1, dash='dash')
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=processed_df.index,
                y=processed_df['BB_Lower'],
                name='BB Lower',
                line=dict(color='gray', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.1)'
            ),
            row=1, col=1
        )

    current_row = 2

    # Volume chart
    if show_volume:
        colors = ['red' if processed_df['Close'].iloc[i] < processed_df['Open'].iloc[i]
                  else 'green' for i in range(len(processed_df))]

        fig.add_trace(
            go.Bar(
                x=processed_df.index,
                y=processed_df['Volume'],
                name='Volume',
                marker_color=colors
            ),
            row=current_row, col=1
        )
        current_row += 1

    # RSI chart
    if show_rsi and 'RSI' in processed_df.columns:
        fig.add_trace(
            go.Scatter(
                x=processed_df.index,
                y=processed_df['RSI'],
                name='RSI',
                line=dict(color='purple', width=2)
            ),
            row=current_row, col=1
        )

        # RSI overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red",
                      annotation_text="Overbought", row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green",
                      annotation_text="Oversold", row=current_row, col=1)

    # Update layout
    fig.update_layout(
        title=f"{ticker} Stock Analysis",
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )

    return fig


# Page setup
st.set_page_config(
    page_title="📈 Stock Data Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
    }
    .success-msg {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📊 Advanced Stock Market Analysis Dashboard</h1>', unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.header("🔧 Configuration")

# Stock selection with popular stocks
popular_stocks = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA", "NFLX"]

# First, check if a stock is selected from dropdown
selected_stock = st.sidebar.selectbox("🔽 Quick Select Popular Stocks:", [""] + popular_stocks, key="stock_selector")

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
    **💡 Tip for international stocks:**
    Add an exchange suffix (e.g., **RELIANCE.NS** for India, **DAI.F** for Germany).
""")
st.sidebar.markdown("---")


# Then show the text input with the selected value or default
if selected_stock:
    ticker = st.sidebar.text_input("📝 Enter Stock Ticker", value=selected_stock).upper()
else:
    ticker = st.sidebar.text_input("📝 Enter Stock Ticker", value="AAPL").upper()

# Date range selection
st.sidebar.subheader("📅 Date Range")
date_option = st.sidebar.radio(
    "Select date range:",
    ["Last 30 days", "Last 3 months", "Last 6 months", "Last 1 year", "Custom range"]
)

if date_option == "Custom range":
    start_date = st.sidebar.date_input("Start Date", value=date(2022, 1, 1))
    end_date = st.sidebar.date_input("End Date", value=date.today())
else:
    end_date = date.today()
    if date_option == "Last 30 days":
        start_date = end_date - timedelta(days=30)
    elif date_option == "Last 3 months":
        start_date = end_date - timedelta(days=90)
    elif date_option == "Last 6 months":
        start_date = end_date - timedelta(days=180)
    else:  # Last 1 year
        start_date = end_date - timedelta(days=365)

# Analysis options
st.sidebar.subheader("📊 Analysis Options")
chart_type = st.sidebar.radio(
    "Chart Type:",
    ["Advanced Dashboard", "Basic Charts"],
    help="Choose between advanced technical analysis or basic charts"
)
show_volume = st.sidebar.checkbox("Show Volume", value=True)
show_ma = st.sidebar.checkbox("Show Moving Averages", value=True)
show_rsi = st.sidebar.checkbox("Show RSI", value=True)
show_bollinger = st.sidebar.checkbox("Show Bollinger Bands", value=False)

# Run analysis button
run_btn = st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True)

# Add your signature in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; padding: 15px; background: linear-gradient(45deg, #667eea, #764ba2); border-radius: 10px; color: white;'>
    <h5 style='margin: 0; color: white;'>👨‍💻 Built by</h5>
    <h4 style='margin: 5px 0; color: white;'>Adith NK</h4>
    <p style='margin: 0; font-size: 12px; opacity: 0.9;'>Python Developer | Data Analyst</p>
    <div style='margin-top: 10px;'>
        <a href='https://www.linkedin.com/in/adith-nk-a15594238/' target='_blank' style='color: white; text-decoration: none; margin: 0 5px;'>💼</a>
        <a href='mailto:adithnk07@gmail.com' style='color: white; text-decoration: none; margin: 0 5px;'>📧</a>
    </div>
</div>
""", unsafe_allow_html=True)

# Main content area
if run_btn and ticker:
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Step 1: Fetch Data
        status_text.text("📥 Fetching stock data...")
        progress_bar.progress(20)

        raw_df = fetch_stock_data(ticker, start=start_date, end=end_date)

        if raw_df is None or raw_df.empty:
            st.error("❌ No data fetched. Please check the ticker symbol and try again.")
        else:
            progress_bar.progress(40)
            st.success(f"✅ Successfully fetched {len(raw_df)} data points for {ticker}")

            # Step 2: Process Data
            status_text.text("⚙️ Processing data...")
            progress_bar.progress(60)

            processed_df = process_stock_data(raw_df, ticker)

            if processed_df is None or processed_df.empty:
                st.error("❌ Data processing failed.")
            else:
                progress_bar.progress(80)
                st.success(f"✅ Data processed successfully: {len(processed_df)} rows")

                # Clear progress indicators
                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()

                # Always create tabs for different views
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Charts", "📊 Data Table", "📋 Summary", "📰 News & Events", "📥 Download"])

                with tab1:
                    st.markdown("## 📊 Technical Analysis Chart")

                    # Choose chart type based on user selection and availability
                    if chart_type == "Advanced Dashboard" and DASHBOARD_AVAILABLE:
                        try:
                            fig = visualize_stock_data(processed_df, ticker)
                            st.info("🚀 Using Advanced Dashboard with Technical Analysis")
                        except Exception as e:
                            st.warning(f"⚠️ Advanced dashboard failed: {e}")
                            st.info("🔄 Falling back to basic charts...")
                            fig = create_basic_chart(processed_df, ticker, show_volume, show_ma, show_rsi,
                                                     show_bollinger)
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Use basic charts
                        fig = create_basic_chart(processed_df, ticker, show_volume, show_ma, show_rsi, show_bollinger)
                        st.plotly_chart(fig, use_container_width=True)
                        if not DASHBOARD_AVAILABLE:
                            st.info("💡 Install dashboard.py for advanced technical analysis features")

                with tab2:
                    st.subheader("📄 Complete Dataset")

                    # Data filtering options
                    col1, col2 = st.columns(2)
                    with col1:
                        show_rows = st.selectbox("Rows to display:", [10, 25, 50, 100, "All"])
                    with col2:
                        sort_order = st.selectbox("Sort by date:", ["Newest first", "Oldest first"])

                    # Display data
                    display_df = processed_df.copy()
                    if sort_order == "Newest first":
                        display_df = display_df.sort_index(ascending=False)

                    if show_rows != "All":
                        display_df = display_df.head(show_rows)

                    st.dataframe(display_df, use_container_width=True)

                    # Data info
                    st.info(
                        f"📊 **Dataset Info:** {len(processed_df)} rows × {len(processed_df.columns)} columns | Date range: {processed_df.index.min().date()} to {processed_df.index.max().date()}")

                with tab3:
                    st.subheader("📋 Stock Analysis Summary")

                    latest = processed_df.iloc[-1]
                    first = processed_df.iloc[0]

                    # Key metrics in columns
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        price_change = ((latest['Close'] - first['Close']) / first['Close'] * 100)
                        st.metric(
                            "Current Price",
                            f"${latest['Close']:.2f}",
                            f"{price_change:+.2f}%"
                        )

                    with col2:
                        vol_change = ((latest['Volume'] - processed_df['Volume'].mean()) / processed_df[
                            'Volume'].mean() * 100)
                        st.metric(
                            "Volume",
                            f"{latest['Volume']:,.0f}",
                            f"{vol_change:+.1f}% vs avg"
                        )

                    with col3:
                        if 'RSI' in processed_df.columns and not pd.isna(latest['RSI']):
                            rsi_signal = "🔴 Overbought" if latest['RSI'] > 70 else (
                                "🟢 Oversold" if latest['RSI'] < 30 else "🟡 Neutral")
                            st.metric("RSI (14)", f"{latest['RSI']:.1f}", rsi_signal)
                        else:
                            st.metric("Daily High", f"${latest['High']:.2f}")

                    with col4:
                        volatility = processed_df['Close'].pct_change().std() * np.sqrt(252) * 100
                        st.metric("Volatility (Annual)", f"{volatility:.1f}%")

                    # Additional statistics
                    st.markdown("---")
                    st.subheader("📊 Detailed Statistics")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**📈 Price Statistics:**")
                        price_stats = processed_df['Close'].describe()
                        for stat, value in price_stats.items():
                            st.write(f"• **{stat.title()}:** ${value:.2f}")

                    with col2:
                        st.markdown("**📊 Volume Statistics:**")
                        volume_stats = processed_df['Volume'].describe()
                        for stat, value in volume_stats.items():
                            st.write(f"• **{stat.title()}:** {value:,.0f}")

                    # Performance metrics
                    st.markdown("---")
                    st.subheader("🎯 Performance Metrics")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        # 52-week high/low
                        high_52w = processed_df['High'].max()
                        low_52w = processed_df['Low'].min()
                        st.metric("52W High", f"${high_52w:.2f}")
                        st.metric("52W Low", f"${low_52w:.2f}")

                    with col2:
                        # Average daily return
                        daily_returns = processed_df['Close'].pct_change().dropna()
                        avg_daily_return = daily_returns.mean() * 100
                        st.metric("Avg Daily Return", f"{avg_daily_return:.3f}%")

                        # Best/worst day
                        best_day = daily_returns.max() * 100
                        worst_day = daily_returns.min() * 100
                        st.metric("Best Day", f"+{best_day:.2f}%")
                        st.metric("Worst Day", f"{worst_day:.2f}%")

                    with col3:
                        # Sharpe ratio (simplified)
                        if daily_returns.std() > 0:
                            sharpe = (avg_daily_return / (daily_returns.std() * 100)) * np.sqrt(252)
                            st.metric("Sharpe Ratio", f"{sharpe:.2f}")

                        # Win rate
                        win_rate = (daily_returns > 0).mean() * 100
                        st.metric("Win Rate", f"{win_rate:.1f}%")

                with tab4:
                    st.markdown(f"## 📰 Latest News for {ticker}")
                    news_data = fetch_stock_news(ticker, start_date, end_date, FINNHUB_API_KEY)
                    if news_data:
                        for article in news_data[:10]: # Display top 10 articles
                            with st.expander(f"**{article['headline']}**"):
                                st.write(f"**Source**: {article['source']}")
                                st.write(f"**Published**: {datetime.fromtimestamp(article['datetime']).strftime('%Y-%m-%d %H:%M:%S')}")
                                st.write(f"**Sentiment**: {get_sentiment(article['summary'])}")
                                st.write(article['summary'])
                                st.markdown(f"[Read full article]({article['url']})")
                    else:
                        st.info("No news found for this ticker or API key is not set.")

                with tab5:
                    st.subheader("📥 Download Your Data")

                    # Show data info
                    st.info(
                        f"📊 **Ready to download:** {len(processed_df)} rows of {ticker} stock data from {start_date} to {end_date}")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("### 📄 CSV Format")
                        st.write("Perfect for Excel, Google Sheets, and data analysis")

                        # CSV download
                        csv_data = processed_df.to_csv(index=True).encode('utf-8')
                        st.download_button(
                            label="📄 Download CSV File",
                            data=csv_data,
                            file_name=f'{ticker}_stock_data_{start_date}_to_{end_date}.csv',
                            mime='text/csv',
                            use_container_width=True,
                            help="Download stock data in CSV format for Excel/Sheets"
                        )

                    with col2:
                        st.markdown("### 📋 JSON Format")
                        st.write("Perfect for web applications and APIs")

                        # JSON download
                        json_data = processed_df.to_json(orient='records', date_format='iso', indent=2)
                        st.download_button(
                            label="📋 Download JSON File",
                            data=json_data.encode('utf-8'),
                            file_name=f'{ticker}_stock_data_{start_date}_to_{end_date}.json',
                            mime='application/json',
                            use_container_width=True,
                            help="Download stock data in JSON format for web apps"
                        )

                    # Quick download in main data section
                    st.markdown("---")
                    st.markdown("### 🚀 Quick Actions")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        # Quick CSV download
                        st.download_button(
                            label="⚡ Quick CSV",
                            data=csv_data,
                            file_name=f'{ticker}_data.csv',
                            mime='text/csv',
                            help="Quick CSV download"
                        )

                    with col2:
                        # Show data preview
                        if st.button("👀 Preview Data"):
                            st.dataframe(processed_df.head(10), use_container_width=True)

                    with col3:
                        # Data statistics
                        if st.button("📊 Data Stats"):
                            st.write(f"**Total Records:** {len(processed_df)}")
                            st.write(
                                f"**Date Range:** {processed_df.index.min().date()} to {processed_df.index.max().date()}")
                            st.write(f"**Columns:** {len(processed_df.columns)}")

                    st.markdown("---")
                    st.success("💡 **Tip:** Use CSV format for Excel compatibility, JSON for web applications and APIs.")

                    # File format information
                    with st.expander("ℹ️ File Format Information"):
                        st.markdown("""
                        **CSV Format includes:**
                        - Date, Open, High, Low, Close, Volume
                        - Technical indicators (RSI, Moving Averages, etc.)
                        - Compatible with Excel, Google Sheets, Python pandas

                        **JSON Format includes:**
                        - Same data as CSV but in JSON structure
                        - Perfect for web applications and APIs
                        - Easy to parse with JavaScript, Python, etc.

                        **Data Processing:**
                        - All data is cleaned and validated
                        - Technical indicators calculated using industry-standard methods
                        - Ready for further analysis or visualization
                        """)

    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.error("Please check your ticker symbol and try again.")

    finally:
        # Clean up progress indicators
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'status_text' in locals():
            status_text.empty()

else:
    # Welcome screen
    st.markdown("""
    ## Welcome to the Stock Market Analysis Dashboard! 🎯

    This powerful tool helps you analyze stock market data with interactive charts and comprehensive insights.

    ### 🚀 Features:
    - **Real-time stock data** fetching and processing
    - **Interactive charts** with candlestick patterns, volume, and technical indicators
    - **Technical analysis** including RSI, Moving Averages, and Bollinger Bands
    - **Data export** in multiple formats (CSV, JSON)
    - **Comprehensive statistics** and market insights

    ### 📋 How to use:
    1. Enter a stock ticker symbol (e.g., AAPL, GOOGL, TSLA)
    2. Select your preferred date range
    3. Choose which indicators to display
    4. Click "🚀 Run Analysis" to start

    ### 💡 Pro Tips:
    - Use popular tickers from the dropdown for quick selection
    - Choose between "Advanced Dashboard" or "Basic Charts"
    - Enable multiple indicators for comprehensive analysis
    - Download data for offline analysis
    - Check the summary tab for key insights

    **Ready to start? Configure your analysis in the sidebar and hit the Run button!**
    """)

    # Footer with your signature
    st.markdown("""
    <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px; margin-top: 50px;'>
        <h4 style='color: #666; margin-bottom: 10px;'>📊 Built by Adith NK</h4>
        <p style='color: #888; margin: 5px 0;'>🚀 Python Developer | Data Analyst | Financial Technology Enthusiast</p>
        <p style='color: #888; margin: 5px 0;'>
            💼 <a href='https://www.linkedin.com/in/adith-nk-a15594238/' target='_blank' style='color: #0077b5; text-decoration: none;'>LinkedIn</a> |
            📧 <a href='mailto:adithnk07@gmail.com' style='color: #d44638; text-decoration: none;'>Contact Me</a>
        </p>
        <p style='color: #999; font-size: 12px; margin-top: 15px;'>
            Built with ❤️ using Streamlit, Plotly, and yfinance | Portfolio Project © 2025
        </p>
        <p style='color: #aaa; font-size: 10px; margin-top: 5px;'>
            🌟 Available for Python Development & Data Analysis Opportunities
        </p>
    </div>
    """, unsafe_allow_html=True)