"""
Enhanced visualization module with interactive candlestick charts
Replace your existing visualize_data.py with this file
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st
import ta
from typing import List, Optional
import warnings

warnings.filterwarnings('ignore')


def calculate_technical_indicators(data: pd.DataFrame) -> dict:
    """
    Calculate all technical indicators

    Args:
        data: DataFrame with OHLCV columns

    Returns:
        Dictionary of calculated indicators
    """
    indicators = {}

    try:
        # Simple Moving Averages
        indicators['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
        indicators['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
        indicators['SMA_200'] = ta.trend.sma_indicator(data['Close'], window=200)

        # Exponential Moving Averages
        indicators['EMA_12'] = ta.trend.ema_indicator(data['Close'], window=12)
        indicators['EMA_26'] = ta.trend.ema_indicator(data['Close'], window=26)

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close=data['Close'])
        indicators['BB_Upper'] = bb.bollinger_hband()
        indicators['BB_Middle'] = bb.bollinger_mavg()
        indicators['BB_Lower'] = bb.bollinger_lband()

        # RSI
        indicators['RSI'] = ta.momentum.rsi(data['Close'])

        # MACD
        macd = ta.trend.MACD(close=data['Close'])
        indicators['MACD'] = macd.macd()
        indicators['MACD_Signal'] = macd.macd_signal()
        indicators['MACD_Histogram'] = macd.macd_diff()

        # Stochastic
        stoch = ta.momentum.StochasticOscillator(
            high=data['High'], low=data['Low'], close=data['Close']
        )
        indicators['Stoch_K'] = stoch.stoch()
        indicators['Stoch_D'] = stoch.stoch_signal()

        # Volume SMA - FIXED VERSION
        try:
            # Try different approaches for Volume SMA
            if hasattr(ta.volume, 'VolumeSMAIndicator'):
                # If class-based approach exists
                vol_sma = ta.volume.VolumeSMAIndicator(close=data['Close'], volume=data['Volume'], window=20)
                indicators['Volume_SMA'] = vol_sma.volume_sma()
            elif hasattr(ta.volume, 'volume_weighted_average_price'):
                # Alternative: use VWAP as proxy
                indicators['Volume_SMA'] = ta.volume.volume_weighted_average_price(
                    high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume']
                )
            else:
                # Manual calculation - most reliable
                indicators['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
        except (AttributeError, Exception) as e:
            # Fallback to manual calculation
            print(f"Volume SMA calculation fallback: {e}")
            indicators['Volume_SMA'] = data['Volume'].rolling(window=20).mean()

        # On Balance Volume
        try:
            indicators['OBV'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])
        except Exception:
            # Manual OBV calculation
            indicators['OBV'] = (data['Volume'] * ((data['Close'] - data['Close'].shift(1)) > 0).astype(int) -
                                 data['Volume'] * ((data['Close'] - data['Close'].shift(1)) < 0).astype(int)).cumsum()

    except Exception as e:
        st.error(f"Error calculating indicators: {str(e)}")
        # Return basic manual calculations as fallback
        indicators = calculate_fallback_indicators(data)

    return indicators


def calculate_fallback_indicators(data: pd.DataFrame) -> dict:
    """
    Fallback function with manual indicator calculations
    """
    indicators = {}

    try:
        # Simple Moving Averages
        indicators['SMA_20'] = data['Close'].rolling(window=20).mean()
        indicators['SMA_50'] = data['Close'].rolling(window=50).mean()
        indicators['SMA_200'] = data['Close'].rolling(window=200).mean()

        # Exponential Moving Averages
        indicators['EMA_12'] = data['Close'].ewm(span=12).mean()
        indicators['EMA_26'] = data['Close'].ewm(span=26).mean()

        # Bollinger Bands
        sma_20 = data['Close'].rolling(window=20).mean()
        std_20 = data['Close'].rolling(window=20).std()
        indicators['BB_Upper'] = sma_20 + (std_20 * 2)
        indicators['BB_Middle'] = sma_20
        indicators['BB_Lower'] = sma_20 - (std_20 * 2)

        # RSI (manual calculation)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        indicators['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        indicators['MACD'] = indicators['EMA_12'] - indicators['EMA_26']
        indicators['MACD_Signal'] = indicators['MACD'].ewm(span=9).mean()
        indicators['MACD_Histogram'] = indicators['MACD'] - indicators['MACD_Signal']

        # Stochastic (manual calculation)
        low_14 = data['Low'].rolling(window=14).min()
        high_14 = data['High'].rolling(window=14).max()
        indicators['Stoch_K'] = 100 * ((data['Close'] - low_14) / (high_14 - low_14))
        indicators['Stoch_D'] = indicators['Stoch_K'].rolling(window=3).mean()

        # Volume SMA (manual)
        indicators['Volume_SMA'] = data['Volume'].rolling(window=20).mean()

        # On Balance Volume (manual)
        indicators['OBV'] = (data['Volume'] * ((data['Close'] - data['Close'].shift(1)) > 0).astype(int) -
                             data['Volume'] * ((data['Close'] - data['Close'].shift(1)) < 0).astype(int)).cumsum()

    except Exception as e:
        print(f"Error in fallback indicators: {e}")
        # Return empty dict if all else fails
        indicators = {}

    return indicators


def create_candlestick_chart(data: pd.DataFrame,
                             symbol: str,
                             selected_indicators: List[str] = None,
                             height: int = 800) -> go.Figure:
    """
    Create interactive candlestick chart with technical indicators

    Args:
        data: Stock data DataFrame with OHLCV columns
        symbol: Stock symbol for title
        selected_indicators: List of indicators to show
        height: Chart height in pixels

    Returns:
        Plotly figure object
    """

    if selected_indicators is None:
        selected_indicators = ['SMA_50', 'SMA_200', 'RSI', 'MACD']

    # Calculate indicators
    indicators = calculate_technical_indicators(data)

    # Create subplots
    rows = 1
    row_heights = [0.6]
    subplot_titles = [f'{symbol} - Stock Price']

    # Add oscillator rows
    if 'RSI' in selected_indicators or 'Stoch_K' in selected_indicators:
        rows += 1
        row_heights.append(0.2)
        subplot_titles.append('Momentum Oscillators')

    if 'MACD' in selected_indicators:
        rows += 1
        row_heights.append(0.2)
        subplot_titles.append('MACD')

    # Always show volume
    rows += 1
    row_heights = [h * 0.8 for h in row_heights] + [0.2]
    subplot_titles.append('Volume')

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
        vertical_spacing=0.03
    )

    # 1. MAIN CANDLESTICK CHART
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='OHLC',
            increasing_line_color='#00C851',
            decreasing_line_color='#ff4444',
            increasing_fillcolor='#00C851',
            decreasing_fillcolor='#ff4444'
        ),
        row=1, col=1
    )

    # 2. MOVING AVERAGES
    ma_colors = {
        'SMA_20': '#FF6B35',
        'SMA_50': '#004E89',
        'SMA_200': '#7209B7',
        'EMA_12': '#F72585',
        'EMA_26': '#4361EE'
    }

    for ma, color in ma_colors.items():
        if ma in selected_indicators and ma in indicators and indicators[ma] is not None:
            try:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=indicators[ma],
                        mode='lines',
                        name=ma.replace('_', ' '),
                        line=dict(color=color, width=2),
                        opacity=0.8
                    ),
                    row=1, col=1
                )
            except Exception as e:
                print(f"Error adding {ma}: {e}")

    # 3. BOLLINGER BANDS
    if 'Bollinger_Bands' in selected_indicators and all(k in indicators for k in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
        try:
            # Upper band
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=indicators['BB_Upper'],
                    mode='lines',
                    line=dict(color='rgba(173, 204, 255, 0.5)', width=1),
                    name='BB Upper',
                    showlegend=False
                ),
                row=1, col=1
            )

            # Lower band with fill
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=indicators['BB_Lower'],
                    mode='lines',
                    line=dict(color='rgba(173, 204, 255, 0.5)', width=1),
                    name='Bollinger Bands',
                    fill='tonexty',
                    fillcolor='rgba(173, 204, 255, 0.1)'
                ),
                row=1, col=1
            )

            # Middle line
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=indicators['BB_Middle'],
                    mode='lines',
                    line=dict(color='rgba(173, 204, 255, 0.8)', dash='dash', width=1),
                    name='BB Middle',
                    showlegend=False
                ),
                row=1, col=1
            )
        except Exception as e:
            print(f"Error adding Bollinger Bands: {e}")

    current_row = 2

    # 4. MOMENTUM OSCILLATORS (RSI, Stochastic)
    if 'RSI' in selected_indicators or 'Stoch_K' in selected_indicators:
        if 'RSI' in selected_indicators and 'RSI' in indicators and indicators['RSI'] is not None:
            try:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=indicators['RSI'],
                        mode='lines',
                        name='RSI',
                        line=dict(color='#FF6B35', width=2)
                    ),
                    row=current_row, col=1
                )

                # RSI levels
                fig.add_hline(y=70, line_dash="dash", line_color="red",
                              row=current_row, col=1, opacity=0.7)
                fig.add_hline(y=30, line_dash="dash", line_color="green",
                              row=current_row, col=1, opacity=0.7)
            except Exception as e:
                print(f"Error adding RSI: {e}")

        if 'Stoch_K' in selected_indicators and 'Stoch_K' in indicators and indicators['Stoch_K'] is not None:
            try:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=indicators['Stoch_K'],
                        mode='lines',
                        name='Stoch %K',
                        line=dict(color='#4361EE', width=2)
                    ),
                    row=current_row, col=1
                )

                if 'Stoch_D' in indicators and indicators['Stoch_D'] is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=indicators['Stoch_D'],
                            mode='lines',
                            name='Stoch %D',
                            line=dict(color='#F72585', width=2)
                        ),
                        row=current_row, col=1
                    )
            except Exception as e:
                print(f"Error adding Stochastic: {e}")

        current_row += 1

    # 5. MACD
    if 'MACD' in selected_indicators and all(k in indicators for k in ['MACD', 'MACD_Signal', 'MACD_Histogram']):
        try:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=indicators['MACD'],
                    mode='lines',
                    name='MACD',
                    line=dict(color='#004E89', width=2)
                ),
                row=current_row, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=indicators['MACD_Signal'],
                    mode='lines',
                    name='MACD Signal',
                    line=dict(color='#FF6B35', width=2)
                ),
                row=current_row, col=1
            )

            # MACD Histogram
            colors = ['green' if val >= 0 else 'red' for val in indicators['MACD_Histogram']]
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=indicators['MACD_Histogram'],
                    name='MACD Histogram',
                    marker_color=colors,
                    opacity=0.6
                ),
                row=current_row, col=1
            )

            current_row += 1
        except Exception as e:
            print(f"Error adding MACD: {e}")
            current_row += 1

    # 6. VOLUME
    try:
        volume_colors = ['green' if close >= open else 'red'
                         for close, open in zip(data['Close'], data['Open'])]

        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color=volume_colors,
                opacity=0.6
            ),
            row=current_row, col=1
        )

        if 'Volume_SMA' in selected_indicators and 'Volume_SMA' in indicators and indicators['Volume_SMA'] is not None:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=indicators['Volume_SMA'],
                    mode='lines',
                    name='Volume SMA',
                    line=dict(color='orange', width=2)
                ),
                row=current_row, col=1
            )
    except Exception as e:
        print(f"Error adding Volume: {e}")

    # UPDATE LAYOUT
    fig.update_layout(
        title=dict(
            text=f"📈 {symbol} - Professional Technical Analysis",
            font=dict(size=24, color='#2c3e50'),
            x=0.5
        ),
        height=height,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=50, r=50, t=80, b=50),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#2c3e50'),
        hovermode='x unified'
    )

    # Remove rangeslider
    fig.update_layout(xaxis_rangeslider_visible=False)

    # Style axes
    fig.update_xaxes(
        gridcolor='#ecf0f1',
        gridwidth=1,
        showgrid=True,
        zeroline=False
    )

    fig.update_yaxes(
        gridcolor='#ecf0f1',
        gridwidth=1,
        showgrid=True,
        zeroline=False
    )

    return fig


def create_indicator_sidebar():
    """Create sidebar controls for indicator selection"""
    st.sidebar.markdown("### 📊 Technical Analysis Controls")

    # Moving Averages
    st.sidebar.markdown("**📈 Trend Indicators**")
    trend_indicators = st.sidebar.multiselect(
        "Moving Averages:",
        ['SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26'],
        default=['SMA_50', 'SMA_200'],
        help="Select moving averages to overlay on price chart"
    )

    bollinger = st.sidebar.checkbox(
        "Bollinger Bands",
        value=False,
        help="Show Bollinger Bands (20-period SMA ± 2 standard deviations)"
    )
    if bollinger:
        trend_indicators.append('Bollinger_Bands')

    # Momentum Oscillators
    st.sidebar.markdown("**⚡ Momentum Oscillators**")
    oscillators = st.sidebar.multiselect(
        "Select Oscillators:",
        ['RSI', 'Stoch_K'],
        default=['RSI'],
        help="RSI: Relative Strength Index, Stoch: Stochastic Oscillator"
    )

    # MACD
    macd_enabled = st.sidebar.checkbox(
        "MACD",
        value=True,
        help="Moving Average Convergence Divergence indicator"
    )
    if macd_enabled:
        oscillators.append('MACD')

    # Volume
    st.sidebar.markdown("**📊 Volume Analysis**")
    volume_sma = st.sidebar.checkbox(
        "Volume SMA",
        value=False,
        help="Show Volume Simple Moving Average"
    )
    if volume_sma:
        trend_indicators.append('Volume_SMA')

    # Chart Settings
    st.sidebar.markdown("**⚙️ Chart Settings**")
    chart_height = st.sidebar.slider(
        "Chart Height",
        600, 1200, 800, 50,
        help="Adjust chart height in pixels"
    )

    all_indicators = trend_indicators + oscillators

    return all_indicators, chart_height


def display_stock_metrics(data: pd.DataFrame, indicators: dict = None):
    """Display key stock metrics in columns"""

    # Calculate basic metrics
    current_price = data['Close'].iloc[-1]
    prev_price = data['Close'].iloc[-2]
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100

    # Volume metrics
    current_volume = data['Volume'].iloc[-1]
    avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

    # 52-week range
    high_52w = data['High'].rolling(252).max().iloc[-1]
    low_52w = data['Low'].rolling(252).min().iloc[-1]
    range_position = (current_price - low_52w) / (high_52w - low_52w) * 100

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "💰 Current Price",
            f"${current_price:.2f}",
            f"{price_change_pct:+.2f}%",
            delta_color="normal"
        )

    with col2:
        st.metric(
            "📊 Volume",
            f"{current_volume:,.0f}",
            f"{((volume_ratio - 1) * 100):+.1f}% vs avg",
            delta_color="normal"
        )

    with col3:
        if indicators and 'RSI' in indicators and indicators['RSI'] is not None:
            try:
                current_rsi = indicators['RSI'].iloc[-1]
                rsi_signal = "🔴 Overbought" if current_rsi > 70 else "🟢 Oversold" if current_rsi < 30 else "🟡 Neutral"
                st.metric(
                    "📈 RSI (14)",
                    f"{current_rsi:.1f}",
                    rsi_signal
                )
            except:
                st.metric(
                    "📊 Daily Range",
                    f"${data['High'].iloc[-1]:.2f}",
                    f"Low: ${data['Low'].iloc[-1]:.2f}"
                )
        else:
            st.metric(
                "📊 Daily Range",
                f"${data['High'].iloc[-1]:.2f}",
                f"Low: ${data['Low'].iloc[-1]:.2f}"
            )

    with col4:
        st.metric(
            "📅 52W Position",
            f"{range_position:.1f}%",
            f"H: ${high_52w:.2f} | L: ${low_52w:.2f}"
        )


# Main function to replace your existing visualization functions
def visualize_stock_data(data: pd.DataFrame, symbol: str):
    """
    Main function to create enhanced stock visualization

    Args:
        data: DataFrame with OHLCV data
        symbol: Stock symbol
    """

    # Create sidebar controls
    selected_indicators, chart_height = create_indicator_sidebar()

    # Create the main chart
    fig = create_candlestick_chart(
        data=data,
        symbol=symbol,
        selected_indicators=selected_indicators,
        height=chart_height
    )

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)

    # Calculate indicators for metrics
    indicators = calculate_technical_indicators(data)

    # Display metrics
    display_stock_metrics(data, indicators)

    return fig


# Backward compatibility functions (keep your existing function names)
def plot_stock_price(data: pd.DataFrame, symbol: str):
    """Backward compatibility wrapper"""
    return visualize_stock_data(data, symbol)


def calculate_sma(data: pd.Series, window: int):
    """Simple moving average calculation"""
    return data.rolling(window=window).mean()


def calculate_ema(data: pd.Series, window: int):
    """Exponential moving average calculation"""
    return data.ewm(span=window).mean()


def calculate_rsi(data: pd.Series, window: int = 14):
    """RSI calculation"""
    try:
        return ta.momentum.rsi(data, window=window)
    except:
        # Manual RSI calculation
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))


# Export the main function
__all__ = ['visualize_stock_data', 'create_candlestick_chart', 'calculate_technical_indicators']
# Add this to the END of your dashboard.py file

if __name__ == "__main__":
    import yfinance as yf
    from datetime import date, timedelta

    # Configure page
    st.set_page_config(
        page_title="📈 Stock Technical Analysis",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Title
    st.title("📊 Advanced Stock Technical Analysis Dashboard")
    st.markdown("---")

    # Sidebar inputs
    st.sidebar.header("🎯 Stock Selection")

    # Stock input
    ticker = st.sidebar.text_input("Enter Stock Symbol:", value="AAPL").upper()

    # Date range
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.sidebar.date_input("Start Date", value=date.today() - timedelta(days=365))
    with col2:
        end_date = st.sidebar.date_input("End Date", value=date.today())

    # Load data button
    if st.sidebar.button("📊 Load Data", type="primary"):
        if ticker:
            try:
                # Fetch data
                with st.spinner(f"Fetching data for {ticker}..."):
                    stock = yf.Ticker(ticker)
                    data = stock.history(start=start_date, end=end_date)

                if not data.empty:
                    st.success(f"✅ Loaded {len(data)} data points for {ticker}")

                    # Create visualization
                    fig = visualize_stock_data(data, ticker)

                    # Display additional info
                    st.markdown("## 📈 Key Metrics")
                    col1, col2, col3, col4 = st.columns(4)

                    current = data['Close'].iloc[-1]
                    prev = data['Close'].iloc[-2]
                    change = ((current - prev) / prev) * 100

                    with col1:
                        st.metric("Current Price", f"${current:.2f}", f"{change:+.2f}%")
                    with col2:
                        st.metric("Volume", f"{data['Volume'].iloc[-1]:,.0f}")
                    with col3:
                        st.metric("High", f"${data['High'].iloc[-1]:.2f}")
                    with col4:
                        st.metric("Low", f"${data['Low'].iloc[-1]:.2f}")

                else:
                    st.error("❌ No data found for this ticker")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("Please enter a stock ticker symbol")

    else:
        # Welcome message
        st.info("👋 Enter a stock ticker in the sidebar and click 'Load Data' to start analysis!")

        st.markdown("""
        ## 🚀 Features:
        - **Interactive Candlestick Charts** with professional styling
        - **Technical Indicators**: Moving Averages, RSI, MACD, Bollinger Bands
        - **Volume Analysis** with color-coded bars
        - **Momentum Oscillators** for trend analysis
        - **Customizable Display** with sidebar controls

        ## 📋 How to Use:
        1. Enter a stock ticker (e.g., AAPL, GOOGL, TSLA)
        2. Select date range
        3. Click "Load Data"
        4. Use sidebar controls to customize indicators

        **Ready to start? Enter a ticker symbol in the sidebar!**
        """)