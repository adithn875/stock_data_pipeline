# src/analyze_data.py
import pandas as pd
import numpy as np
import os
from typing import Dict, Any, List
import json


def analyze_stock_data(ticker: str) -> Dict[str, Any]:
    """
    Perform comprehensive analysis on processed stock data

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dictionary containing analysis results
    """
    try:
        # Load processed data
        processed_file = f"data/processed/{ticker.replace('.', '_')}_processed.csv"
        if not os.path.exists(processed_file):
            raise FileNotFoundError(f"Processed data file not found: {processed_file}")

        print(f"📈 Analyzing data for: {ticker}")
        df = pd.read_csv(processed_file, index_col=0, parse_dates=True)

        # Perform various analyses
        analysis_results = {
            'ticker': ticker,
            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_summary': get_basic_stats(df),
            'technical_analysis': perform_technical_analysis(df),
            'trend_analysis': perform_trend_analysis(df),
            'volatility_analysis': perform_volatility_analysis(df),
            'volume_analysis': perform_volume_analysis(df),
            'support_resistance': find_support_resistance_levels(df),
            'signals': generate_trading_signals(df),
            'risk_metrics': calculate_risk_metrics(df)
        }

        # Save analysis results
        analysis_file = f"data/processed/{ticker.replace('.', '_')}_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)

        print(f"✅ Analysis saved to: {analysis_file}")

        # Print summary
        print_analysis_summary(analysis_results)

        return analysis_results

    except Exception as e:
        print(f"❌ Error analyzing data for {ticker}: {str(e)}")
        raise


def get_basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Get basic statistical information"""
    current_price = df['Close'].iloc[-1]
    previous_close = df['Close'].iloc[-2] if len(df) > 1 else current_price

    return {
        'total_days': len(df),
        'date_range': {
            'start': df.index.min().strftime('%Y-%m-%d'),
            'end': df.index.max().strftime('%Y-%m-%d')
        },
        'current_price': round(current_price, 2),
        'previous_close': round(previous_close, 2),
        'daily_change': round(current_price - previous_close, 2),
        'daily_change_pct': round(((current_price / previous_close) - 1) * 100, 2),
        'period_high': round(df['High'].max(), 2),
        'period_low': round(df['Low'].min(), 2),
        'period_return': round(((current_price / df['Close'].iloc[0]) - 1) * 100, 2),
        'avg_volume': int(df['Volume'].mean()),
        'total_volume': int(df['Volume'].sum())
    }


def perform_technical_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform technical indicator analysis"""
    current_data = df.iloc[-1]

    # Moving average analysis
    ma_analysis = {
        'sma_10': round(current_data.get('SMA_10', 0), 2),
        'sma_20': round(current_data.get('SMA_20', 0), 2),
        'sma_50': round(current_data.get('SMA_50', 0), 2),
        'price_vs_sma10': 'Above' if current_data['Close'] > current_data.get('SMA_10', 0) else 'Below',
        'price_vs_sma20': 'Above' if current_data['Close'] > current_data.get('SMA_20', 0) else 'Below',
        'price_vs_sma50': 'Above' if current_data['Close'] > current_data.get('SMA_50', 0) else 'Below'
    }

    # RSI analysis
    current_rsi = current_data.get('RSI', 50)
    rsi_condition = 'Overbought' if current_rsi > 70 else 'Oversold' if current_rsi < 30 else 'Neutral'

    # MACD analysis
    macd_signal = 'Bullish' if current_data.get('MACD', 0) > current_data.get('MACD_Signal', 0) else 'Bearish'

    # Bollinger Bands analysis
    bb_position = current_data.get('BB_Position', 0.5)
    bb_signal = 'Upper Band' if bb_position > 0.8 else 'Lower Band' if bb_position < 0.2 else 'Middle Range'

    return {
        'moving_averages': ma_analysis,
        'rsi': {
            'value': round(current_rsi, 2),
            'condition': rsi_condition
        },
        'macd': {
            'value': round(current_data.get('MACD', 0), 4),
            'signal': round(current_data.get('MACD_Signal', 0), 4),
            'histogram': round(current_data.get('MACD_Histogram', 0), 4),
            'trend': macd_signal
        },
        'bollinger_bands': {
            'upper': round(current_data.get('BB_Upper', 0), 2),
            'middle': round(current_data.get('BB_Middle', 0), 2),
            'lower': round(current_data.get('BB_Lower', 0), 2),
            'position': round(bb_position, 3),
            'signal': bb_signal
        }
    }


def perform_trend_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze price trends"""
    recent_data = df.tail(20)  # Last 20 days

    # Calculate trend slopes
    short_term_slope = calculate_trend_slope(recent_data['Close'].tail(5))
    medium_term_slope = calculate_trend_slope(recent_data['Close'].tail(10))
    long_term_slope = calculate_trend_slope(recent_data['Close'])

    # Determine trend direction
    def get_trend_direction(slope):
        if slope > 0.001:
            return 'Uptrend'
        elif slope < -0.001:
            return 'Downtrend'
        else:
            return 'Sideways'

    # Moving average trend
    sma_trend = 'Bullish' if df['SMA_10'].iloc[-1] > df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1] else 'Bearish'

    return {
        'short_term': {
            'slope': round(short_term_slope, 6),
            'direction': get_trend_direction(short_term_slope)
        },
        'medium_term': {
            'slope': round(medium_term_slope, 6),
            'direction': get_trend_direction(medium_term_slope)
        },
        'long_term': {
            'slope': round(long_term_slope, 6),
            'direction': get_trend_direction(long_term_slope)
        },
        'ma_alignment': sma_trend
    }


def perform_volatility_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze price volatility"""
    returns = df['Daily_Return'].dropna()

    return {
        'daily_volatility': round(returns.std(), 4),
        'annualized_volatility': round(returns.std() * np.sqrt(252), 4),
        'volatility_10d': round(df['Volatility_10'].iloc[-1], 4) if 'Volatility_10' in df.columns else 0,
        'volatility_30d': round(df['Volatility_30'].iloc[-1], 4) if 'Volatility_30' in df.columns else 0,
        'max_drawdown': round(calculate_max_drawdown(df['Close']), 4),
        'sharpe_ratio': round(calculate_sharpe_ratio(returns), 4),
        'var_95': round(returns.quantile(0.05), 4),  # Value at Risk (95%)
        'volatility_regime': get_volatility_regime(df)
    }


def perform_volume_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze trading volume patterns"""
    recent_volume = df['Volume'].tail(10).mean()
    avg_volume = df['Volume'].mean()
    volume_ratio = recent_volume / avg_volume

    volume_trend = 'Increasing' if volume_ratio > 1.2 else 'Decreasing' if volume_ratio < 0.8 else 'Stable'

    return {
        'current_volume': int(df['Volume'].iloc[-1]),
        'average_volume': int(avg_volume),
        'recent_avg_volume': int(recent_volume),
        'volume_ratio': round(volume_ratio, 2),
        'volume_trend': volume_trend,
        'volume_price_correlation': round(df['Volume'].corr(df['Close']), 3),
        'high_volume_days': int((df['Volume'] > df['Volume'].quantile(0.9)).sum())
    }


def find_support_resistance_levels(df: pd.DataFrame, window: int = 20) -> Dict[str, List[float]]:
    """Find potential support and resistance levels"""
    highs = df['High'].rolling(window=window, center=True).max()
    lows = df['Low'].rolling(window=window, center=True).min()

    # Find peaks and troughs
    resistance_levels = []
    support_levels = []

    for i in range(window, len(df) - window):
        if df['High'].iloc[i] == highs.iloc[i]:
            resistance_levels.append(df['High'].iloc[i])
        if df['Low'].iloc[i] == lows.iloc[i]:
            support_levels.append(df['Low'].iloc[i])

    # Get unique levels and sort
    resistance_levels = sorted(list(set([round(level, 2) for level in resistance_levels[-10:]])), reverse=True)
    support_levels = sorted(list(set([round(level, 2) for level in support_levels[-10:]])))

    return {
        'resistance_levels': resistance_levels[:5],  # Top 5
        'support_levels': support_levels[-5:]  # Bottom 5
    }


def generate_trading_signals(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate basic trading signals"""
    current = df.iloc[-1]
    previous = df.iloc[-2] if len(df) > 1 else current

    signals = []
    signal_strength = 0

    # RSI signals
    if current.get('RSI', 50) < 30:
        signals.append("RSI Oversold - Potential Buy")
        signal_strength += 1
    elif current.get('RSI', 50) > 70:
        signals.append("RSI Overbought - Potential Sell")
        signal_strength -= 1

    # MACD signals
    if current.get('MACD', 0) > current.get('MACD_Signal', 0) and previous.get('MACD', 0) <= previous.get('MACD_Signal',
                                                                                                          0):
        signals.append("MACD Bullish Crossover")
        signal_strength += 1
    elif current.get('MACD', 0) < current.get('MACD_Signal', 0) and previous.get('MACD', 0) >= previous.get(
            'MACD_Signal', 0):
        signals.append("MACD Bearish Crossover")
        signal_strength -= 1

    # Moving average signals
    if current['Close'] > current.get('SMA_20', 0) and previous['Close'] <= previous.get('SMA_20', 0):
        signals.append("Price broke above SMA20")
        signal_strength += 1
    elif current['Close'] < current.get('SMA_20', 0) and previous['Close'] >= previous.get('SMA_20', 0):
        signals.append("Price broke below SMA20")
        signal_strength -= 1

    # Volume signals
    if current.get('Volume_Ratio', 1) > 1.5:
        signals.append("High volume activity")
        signal_strength += 0.5

    overall_signal = 'Bullish' if signal_strength > 1 else 'Bearish' if signal_strength < -1 else 'Neutral'

    return {
        'signals': signals,
        'signal_strength': round(signal_strength, 1),
        'overall_signal': overall_signal,
        'confidence': min(abs(signal_strength) * 20, 100)  # Convert to percentage
    }


def calculate_risk_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate various risk metrics"""
    returns = df['Daily_Return'].dropna()

    return {
        'beta': calculate_beta(returns),  # Simplified beta calculation
        'alpha': calculate_alpha(returns),
        'information_ratio': calculate_information_ratio(returns),
        'sortino_ratio': calculate_sortino_ratio(returns),
        'calmar_ratio': calculate_calmar_ratio(df['Close'], returns),
        'maximum_drawdown': round(calculate_max_drawdown(df['Close']), 4),
        'recovery_time': calculate_recovery_time(df['Close'])
    }


# Helper functions
def calculate_trend_slope(prices: pd.Series) -> float:
    """Calculate trend slope using linear regression"""
    x = np.arange(len(prices))
    coeffs = np.polyfit(x, prices, 1)
    return coeffs[0] / prices.mean()  # Normalized slope


def calculate_max_drawdown(prices: pd.Series) -> float:
    """Calculate maximum drawdown"""
    peak = prices.expanding().max()
    drawdown = (prices - peak) / peak
    return drawdown.min()


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio"""
    excess_returns = returns.mean() * 252 - risk_free_rate
    volatility = returns.std() * np.sqrt(252)
    return excess_returns / volatility if volatility != 0 else 0


def calculate_beta(returns: pd.Series) -> float:
    """Simplified beta calculation (assumes market return is 8% annually)"""
    market_return = 0.08 / 252  # Daily market return
    covariance = returns.cov(pd.Series([market_return] * len(returns)))
    market_variance = (market_return ** 2)
    return covariance / market_variance if market_variance != 0 else 1


def calculate_alpha(returns: pd.Series) -> float:
    """Calculate alpha"""
    annual_return = returns.mean() * 252
    market_return = 0.08  # Assumed market return
    beta = calculate_beta(returns)
    return annual_return - (0.02 + beta * (market_return - 0.02))  # Risk-free rate = 2%


def calculate_information_ratio(returns: pd.Series) -> float:
    """Calculate information ratio"""
    benchmark_return = 0.08 / 252  # Daily benchmark return
    excess_returns = returns - benchmark_return
    tracking_error = excess_returns.std()
    return excess_returns.mean() / tracking_error if tracking_error != 0 else 0


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sortino ratio"""
    excess_returns = returns.mean() * 252 - risk_free_rate
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252)
    return excess_returns / downside_deviation if downside_deviation != 0 else 0


def calculate_calmar_ratio(prices: pd.Series, returns: pd.Series) -> float:
    """Calculate Calmar ratio"""
    annual_return = returns.mean() * 252
    max_drawdown = abs(calculate_max_drawdown(prices))
    return annual_return / max_drawdown if max_drawdown != 0 else 0


def calculate_recovery_time(prices: pd.Series) -> int:
    """Calculate time to recover from maximum drawdown"""
    peak = prices.expanding().max()
    drawdown = (prices - peak) / peak
    max_dd_idx = drawdown.idxmin()
    recovery_prices = prices.loc[max_dd_idx:]
    max_dd_peak = peak.loc[max_dd_idx]

    recovery_idx = recovery_prices[recovery_prices >= max_dd_peak].index
    if len(recovery_idx) > 0:
        return (recovery_idx[0] - max_dd_idx).days
    else:
        return -1  # Not yet recovered


def get_volatility_regime(df: pd.DataFrame) -> str:
    """Determine current volatility regime"""
    current_vol = df['Volatility_30'].iloc[-1] if 'Volatility_30' in df.columns else 0
    historical_vol = df['Daily_Return'].std() * np.sqrt(252)

    if current_vol > historical_vol * 1.5:
        return 'High Volatility'
    elif current_vol < historical_vol * 0.7:
        return 'Low Volatility'
    else:
        return 'Normal Volatility'


def print_analysis_summary(analysis: Dict[str, Any]):
    """Print a summary of the analysis results"""
    print("\n" + "=" * 60)
    print(f"📊 ANALYSIS SUMMARY FOR {analysis['ticker']}")
    print("=" * 60)

    # Basic stats
    stats = analysis['data_summary']
    print(f"💰 Current Price: ${stats['current_price']}")
    print(f"📈 Daily Change: {stats['daily_change']} ({stats['daily_change_pct']}%)")
    print(f"📊 Period Return: {stats['period_return']}%")
    print(f"🔺 Period High: ${stats['period_high']}")
    print(f"🔻 Period Low: ${stats['period_low']}")

    # Technical indicators
    tech = analysis['technical_analysis']
    print(f"\n🔧 Technical Indicators:")
    print(f"   RSI: {tech['rsi']['value']} ({tech['rsi']['condition']})")
    print(f"   MACD: {tech['macd']['trend']}")
    print(f"   BB Position: {tech['bollinger_bands']['signal']}")

    # Signals
    signals = analysis['signals']
    print(f"\n🎯 Trading Signals:")
    print(f"   Overall: {signals['overall_signal']} (Confidence: {signals['confidence']}%)")
    for signal in signals['signals'][:3]:  # Show top 3 signals
        print(f"   • {signal}")

    # Risk metrics
    risk = analysis['risk_metrics']
    print(f"\n⚠️ Risk Metrics:")
    print(f"   Max Drawdown: {risk['maximum_drawdown'] * 100:.2f}%")
    print(f"   Sharpe Ratio: {risk['sharpe_ratio']:.2f}")
    print(f"   Beta: {risk['beta']:.2f}")

    print("=" * 60)


if __name__ == "__main__":
    # Test the analysis function
    test_ticker = "AAPL"
    try:
        results = analyze_stock_data(test_ticker)
        print(f"✅ Successfully analyzed data for {test_ticker}")
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")