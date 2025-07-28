# fetch_data.py - Improved version with better connection handling
import yfinance as yf
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


def test_connection():
    """Test internet connectivity and Yahoo Finance accessibility"""
    print("🔗 Testing internet connection...")

    # Test basic internet connectivity
    try:
        response = requests.get("https://www.google.com", timeout=10)
        if response.status_code == 200:
            print("✅ Internet connection: OK")
        else:
            print("❌ Internet connection: Failed")
            return False
    except Exception as e:
        print(f"❌ Internet connection: Failed - {str(e)}")
        return False

    # Test Yahoo Finance with proper headers
    print("🔗 Testing Yahoo Finance accessibility...")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get("https://finance.yahoo.com", headers=headers, timeout=15)
        if response.status_code == 200:
            print("✅ Yahoo Finance: Accessible")
            return True
        else:
            print(f"❌ Yahoo Finance: HTTP {response.status_code}")
            return False
    except requests.exceptions.SSLError:
        print("❌ Yahoo Finance: SSL Certificate error")
        return False
    except requests.exceptions.Timeout:
        print("❌ Yahoo Finance: Connection timeout")
        return False
    except Exception as e:
        print(f"❌ Yahoo Finance: {str(e)}")
        return False


def robust_fetch_stock_data(ticker, start=None, end=None, max_retries=3):
    """
    Robust stock data fetching with multiple fallback strategies
    """
    print(f"🎯 Starting robust fetch for {ticker}")

    # Test connection first
    if not test_connection():
        print("❌ Connection issues detected. Trying alternative methods...")
        # Continue anyway - sometimes yfinance works even when direct requests fail

    # Set default date range if not provided
    if end is None:
        end = datetime.now()
    if start is None:
        start = end - timedelta(days=365)  # Default to 1 year

    # Convert to string format if datetime objects
    if isinstance(start, datetime):
        start = start.strftime('%Y-%m-%d')
    if isinstance(end, datetime):
        end = end.strftime('%Y-%m-%d')

    print(f"📅 Fetching data from {start} to {end}")

    # Strategy 1: Try yfinance with simple approach (API change fix)
    for attempt in range(max_retries):
        try:
            print(f"📡 Attempt {attempt + 1}: Using yfinance (simple method)...")

            # Create ticker object without session (yfinance handles this now)
            stock = yf.Ticker(ticker)

            # Fetch data
            data = stock.history(start=start, end=end, auto_adjust=True, prepost=True)

            if not data.empty:
                print(f"✅ Successfully fetched {len(data)} rows of data")

                # Clean and validate data
                data = data.dropna()
                if len(data) > 0:
                    # Reset index to make Date a column
                    data = data.reset_index()
                    print(f"📊 Data columns: {list(data.columns)}")
                    print(f"📊 Date range: {data.Date.min()} to {data.Date.max()}")
                    return data

            print(f"⚠️ Attempt {attempt + 1} returned empty data")

        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print(f"⏳ Waiting 2 seconds before retry...")
                time.sleep(2)

    # Strategy 2: Try without session
    print("📡 Trying yfinance without session...")
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start, end=end, auto_adjust=True)

        if not data.empty:
            data = data.dropna().reset_index()
            print(f"✅ Fallback method successful: {len(data)} rows")
            return data
    except Exception as e:
        print(f"❌ Fallback method failed: {str(e)}")

    # Strategy 3: Try with different date format
    print("📡 Trying with period instead of dates...")
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1y", auto_adjust=True)  # Default to 1 year

        if not data.empty:
            data = data.dropna().reset_index()
            print(f"✅ Period-based fetch successful: {len(data)} rows")
            return data
    except Exception as e:
        print(f"❌ Period-based fetch failed: {str(e)}")

    # Strategy 4: Try basic info to test ticker validity
    print("📡 Testing ticker validity...")
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if info and 'symbol' in info:
            print(f"✅ Ticker {ticker} is valid")
            print("❌ But unable to fetch historical data")
        else:
            print(f"❌ Ticker {ticker} may be invalid")
    except Exception as e:
        print(f"❌ Ticker validation failed: {str(e)}")

    print("❌ All fetch strategies failed")
    return None


def fetch_stock_data_simple(ticker, period="1y"):
    """
    Simplified version for testing
    """
    try:
        print(f"🔍 Simple fetch for {ticker} (period: {period})")
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)

        if not data.empty:
            data = data.reset_index()
            print(f"✅ Simple fetch successful: {len(data)} rows")
            return data
        else:
            print("❌ Simple fetch returned empty data")
            return None
    except Exception as e:
        print(f"❌ Simple fetch failed: {str(e)}")
        return None


# Test function
if __name__ == "__main__":
    print("🧪 Testing stock data fetch...")

    # Test with AAPL
    data = robust_fetch_stock_data("AAPL")
    if data is not None:
        print(f"✅ Test successful! Got {len(data)} rows")
        print("📊 Sample data:")
        print(data.head())
    else:
        print("❌ Test failed - trying simple method")
        data = fetch_stock_data_simple("AAPL")
        if data is not None:
            print(f"✅ Simple method worked! Got {len(data)} rows")
        else:
            print("❌ All methods failed")