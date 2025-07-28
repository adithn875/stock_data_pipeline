# src/store_data.py - Fixed version without Utils dependency
import os
import pandas as pd
import logging
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def store_processed_data(df: pd.DataFrame, ticker: str, output_dir: str = "data/processed") -> bool:
    """
    Store processed stock data to CSV file

    Args:
        df: Processed DataFrame containing stock data and indicators
        ticker: Stock ticker symbol
        output_dir: Directory to save the processed data

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Clean ticker symbol for filename (replace dots with underscores)
        clean_ticker = ticker.replace('.', '_')

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"{clean_ticker}.csv"
        filepath = os.path.join(output_dir, filename)

        # Save DataFrame to CSV
        df.to_csv(filepath, index=False)

        # Log success
        logger.info(f"✅ Processed data saved: {filepath}")
        logger.info(f"   Records: {len(df)}")
        logger.info(f"   Columns: {len(df.columns)}")

        print(f"ℹ️  Data successfully stored at: {filepath}")

        return True

    except Exception as e:
        error_msg = f"Failed to store data for {ticker}: {str(e)}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")
        return False


def store_raw_data(df: pd.DataFrame, ticker: str, output_dir: str = "data/raw") -> bool:
    """
    Store raw stock data to CSV file

    Args:
        df: Raw DataFrame from yfinance
        ticker: Stock ticker symbol
        output_dir: Directory to save the raw data

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Clean ticker symbol for filename
        clean_ticker = ticker.replace('.', '_')
        filename = f"{clean_ticker}_raw.csv"
        filepath = os.path.join(output_dir, filename)

        # Save DataFrame to CSV with index (Date)
        df.to_csv(filepath, index=True)

        logger.info(f"✅ Raw data saved: {filepath}")
        print(f"ℹ️  Raw data stored at: {filepath}")

        return True

    except Exception as e:
        error_msg = f"Failed to store raw data for {ticker}: {str(e)}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")
        return False


def load_processed_data(ticker: str, data_dir: str = "data/processed") -> Optional[pd.DataFrame]:
    """
    Load processed stock data from CSV file

    Args:
        ticker: Stock ticker symbol
        data_dir: Directory containing the processed data

    Returns:
        DataFrame if successful, None otherwise
    """
    try:
        clean_ticker = ticker.replace('.', '_')
        filename = f"{clean_ticker}.csv"
        filepath = os.path.join(data_dir, filename)

        if not os.path.exists(filepath):
            logger.warning(f"File not found: {filepath}")
            return None

        # Load DataFrame
        df = pd.read_csv(filepath)

        # Convert Date column to datetime if it exists
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])

        logger.info(f"✅ Loaded processed data: {filepath} ({len(df)} rows)")
        return df

    except Exception as e:
        error_msg = f"Failed to load processed data for {ticker}: {str(e)}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")
        return None


def get_available_tickers(data_dir: str = "data/processed") -> list:
    """
    Get list of available ticker symbols from processed data directory

    Args:
        data_dir: Directory containing processed data files

    Returns:
        List of available ticker symbols
    """
    try:
        if not os.path.exists(data_dir):
            return []

        tickers = []
        for filename in os.listdir(data_dir):
            if filename.endswith('.csv') and not filename.endswith('_raw.csv'):
                # Extract ticker from filename
                ticker = filename.replace('.csv', '').replace('_', '.')
                tickers.append(ticker)

        return sorted(tickers)

    except Exception as e:
        logger.error(f"Failed to get available tickers: {str(e)}")
        return []


def create_summary_report(data_dir: str = "data/processed") -> bool:
    """
    Create a summary report of all processed stocks

    Args:
        data_dir: Directory containing processed data files

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        tickers = get_available_tickers(data_dir)

        if not tickers:
            print("No processed data found")
            return False

        summary_data = []

        for ticker in tickers:
            df = load_processed_data(ticker, data_dir)
            if df is not None and not df.empty:
                latest = df.iloc[-1]
                summary_data.append({
                    'Ticker': ticker,
                    'Latest_Date': latest.get('Date', 'N/A'),
                    'Latest_Close': latest.get('Close', 'N/A'),
                    'Latest_Volume': latest.get('Volume', 'N/A'),
                    'Latest_RSI': latest.get('RSI', 'N/A'),
                    'Records_Count': len(df)
                })

        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_data)

        # Save summary report
        summary_path = os.path.join(data_dir, 'summary_report.csv')
        summary_df.to_csv(summary_path, index=False)

        logger.info(f"✅ Summary report created: {summary_path}")
        print(f"📊 Summary report created: {summary_path}")

        return True

    except Exception as e:
        error_msg = f"Failed to create summary report: {str(e)}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")
        return False