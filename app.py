# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import date, datetime, timedelta
import numpy as np

# Import your modular pipeline functions (fixed import paths)
from fetch_data import robust_fetch_stock_data as fetch_stock_data
from process_data import process_stock_data
from visualize_data import visualize_stock_data

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

                # Create tabs for different views
                tab1, tab2, tab3, tab4 = st.tabs(["📈 Charts", "📊 Data Table", "📋 Summary", "📥 Download"])

                with tab1:
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
                        title=f"{ticker} Stock Analysis - {start_date} to {end_date}",
                        height=800,
                        showlegend=True,
                        xaxis_rangeslider_visible=False
                    )

                    st.plotly_chart(fig, use_container_width=True)

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

                with tab3:
                    st.subheader("📋 Stock Analysis Summary")

                    latest = processed_df.iloc[-1]
                    first = processed_df.iloc[0]

                    # Key metrics in columns
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric(
                            "Current Price",
                            f"${latest['Close']:.2f}",
                            f"{((latest['Close'] - first['Close']) / first['Close'] * 100):+.2f}%"
                        )

                    with col2:
                        st.metric(
                            "Volume",
                            f"{latest['Volume']:,.0f}",
                            f"{((latest['Volume'] - processed_df['Volume'].mean()) / processed_df['Volume'].mean() * 100):+.1f}%"
                        )

                    with col3:
                        if 'RSI' in latest:
                            rsi_signal = "Overbought" if latest['RSI'] > 70 else (
                                "Oversold" if latest['RSI'] < 30 else "Neutral")
                            st.metric("RSI", f"{latest['RSI']:.1f}", rsi_signal)

                    with col4:
                        volatility = processed_df['Close'].pct_change().std() * np.sqrt(252) * 100
                        st.metric("Volatility (Annual)", f"{volatility:.1f}%")

                    # Additional statistics
                    st.subheader("📊 Statistical Summary")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**Price Statistics:**")
                        price_stats = processed_df['Close'].describe()
                        for stat, value in price_stats.items():
                            st.write(f"• {stat.title()}: ${value:.2f}")

                    with col2:
                        st.write("**Volume Statistics:**")
                        volume_stats = processed_df['Volume'].describe()
                        for stat, value in volume_stats.items():
                            st.write(f"• {stat.title()}: {value:,.0f}")

                with tab4:
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
        st.error("Please check your data pipeline functions and try again.")

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
    - Enable multiple indicators for comprehensive analysis
    - Download data for offline analysis
    - Check the summary tab for key insights

    **Ready to start? Configure your analysis in the sidebar and hit the Run button!**
    """)

    # Sample charts or recent market data could go here
    st.markdown("---")

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