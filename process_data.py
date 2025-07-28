# src/process_data.py
import pandas as pd
import numpy as np
from typing import Optional
import logging
from ta.trend import SMAIndicator
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator

logger = logging.getLogger(__name__)


def process_stock_data(raw_df: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
    """Final version with guaranteed type safety"""
    try:
        # Enhanced input validation
        if raw_df is None:
            raise ValueError(f"raw_df is None for {ticker}")

        if not isinstance(raw_df, pd.DataFrame):
            raise ValueError(f"raw_df is not a DataFrame for {ticker}, got {type(raw_df)}")

        if raw_df.empty:
            raise ValueError(f"raw_df is empty for {ticker}")

        logger.info(f"Processing data for {ticker}: {len(raw_df)} rows")
        logger.debug(f"Raw data columns: {list(raw_df.columns)}")
        logger.debug(f"Raw data shape: {raw_df.shape}")

        # Create working copy
        df = raw_df.copy()

        # Verify required columns exist first
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in numeric_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for {ticker}: {missing_cols}")

        # Clean and convert numeric columns with better error handling
        for col in numeric_cols:
            if col in df.columns:
                try:
                    # Convert to numeric, coercing errors to NaN
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                    # Check if we lost all data
                    if df[col].isna().all():
                        raise ValueError(f"All values in column {col} are invalid for {ticker}")

                    # Keep as pandas Series - DO NOT convert to numpy array
                    df[col] = df[col].astype('float64')

                except Exception as e:
                    raise ValueError(f"Failed to convert column {col} for {ticker}: {str(e)}")

        # Ensure we have valid Close prices
        if 'Close' not in df.columns or df['Close'].isna().all():
            raise ValueError(f"No valid Close prices for {ticker}")

        # Get Close prices as proper Series with validation
        try:
            close_prices = df['Close'].copy()
            if close_prices.empty or close_prices.isna().all():
                raise ValueError(f"Close prices are empty or all NaN for {ticker}")
        except Exception as e:
            raise ValueError(f"Failed to extract Close prices for {ticker}: {str(e)}")

        # Add indicators with enhanced error handling
        df = _add_technical_indicators(df, close_prices, ticker)
        df['Ticker'] = ticker
        df = _add_date_features(df)
        df = _clean_processed_data(df)

        if not _validate_processed_data(df, ticker):
            raise ValueError(f"Validation failed for {ticker}")

        logger.info(f"Successfully processed {ticker}")
        return df

    except Exception as e:
        logger.error(f"Error processing {ticker}: {str(e)}")
        logger.debug(f"Raw data info for {ticker}: shape={getattr(raw_df, 'shape', 'N/A')}, type={type(raw_df)}")
        return None


def _add_technical_indicators(df: pd.DataFrame, close_prices: pd.Series, ticker: str) -> pd.DataFrame:
    """Add indicators with absolute type safety"""
    try:
        # Validate close_prices
        if close_prices is None or close_prices.empty:
            raise ValueError(f"Invalid close_prices for {ticker}")

        # Ensure it's a proper Series - DO NOT convert to numpy array
        if not isinstance(close_prices, pd.Series):
            try:
                close_prices = pd.Series(close_prices, index=df.index)
            except Exception as e:
                raise ValueError(f"Cannot convert close_prices to Series for {ticker}: {str(e)}")

        # Remove any NaN values for indicator calculations
        valid_data_mask = ~close_prices.isna()
        if valid_data_mask.sum() < 50:  # Need minimum data for indicators
            logger.warning(f"Insufficient valid data for indicators for {ticker}")
            # Add empty indicator columns
            indicator_cols = ['SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal',
                              'RSI', 'BB_Upper', 'BB_Middle', 'BB_Lower']
            for col in indicator_cols:
                df[col] = np.nan
            return df

        # Moving Averages with error handling
        try:
            df['SMA_20'] = SMAIndicator(close=close_prices, window=20).sma_indicator()
        except Exception as e:
            logger.warning(f"SMA_20 calculation failed for {ticker}: {str(e)}")
            df['SMA_20'] = np.nan

        try:
            df['SMA_50'] = SMAIndicator(close=close_prices, window=50).sma_indicator()
        except Exception as e:
            logger.warning(f"SMA_50 calculation failed for {ticker}: {str(e)}")
            df['SMA_50'] = np.nan

        # MACD with error handling
        try:
            df['EMA_12'] = close_prices.ewm(span=12, adjust=False).mean()
            df['EMA_26'] = close_prices.ewm(span=26, adjust=False).mean()
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        except Exception as e:
            logger.warning(f"MACD calculation failed for {ticker}: {str(e)}")
            df['EMA_12'] = df['EMA_26'] = df['MACD'] = df['MACD_Signal'] = np.nan

        # RSI with error handling
        try:
            df['RSI'] = RSIIndicator(close=close_prices, window=14).rsi()
        except Exception as e:
            logger.warning(f"RSI calculation failed for {ticker}: {str(e)}")
            df['RSI'] = np.nan

        # Bollinger Bands with error handling
        try:
            bb = BollingerBands(close=close_prices, window=20, window_dev=2)
            df['BB_Upper'] = bb.bollinger_hband()
            df['BB_Middle'] = bb.bollinger_mavg()
            df['BB_Lower'] = bb.bollinger_lband()
        except Exception as e:
            logger.warning(f"Bollinger Bands calculation failed for {ticker}: {str(e)}")
            df['BB_Upper'] = df['BB_Middle'] = df['BB_Lower'] = np.nan

        return df
    except Exception as e:
        raise ValueError(f"Indicator calculation failed for {ticker}: {str(e)}")


def _add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add date-based features"""
    try:
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        df['Year'] = df.index.year
        df['Month'] = df.index.month
        df['Day'] = df.index.day
        df['DayOfWeek'] = df.index.dayofweek
        df['Quarter'] = df.index.quarter

        return df
    except Exception as e:
        logger.warning(f"Date feature extraction failed: {str(e)}")
        # Add empty columns if date processing fails
        df['Year'] = df['Month'] = df['Day'] = df['DayOfWeek'] = df['Quarter'] = np.nan
        return df


def _clean_processed_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the processed data"""
    try:
        # Remove rows where all price data is NaN
        price_cols = ['Open', 'High', 'Low', 'Close']
        df = df.dropna(subset=price_cols, how='all')

        # Forward fill and backward fill for small gaps (updated syntax)
        df = df.ffill(limit=3)
        df = df.bfill(limit=3)

        # Reset index to make Date a column
        df = df.reset_index()
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])

        return df
    except Exception as e:
        logger.warning(f"Data cleaning failed: {str(e)}")
        return df


def _validate_processed_data(df: pd.DataFrame, ticker: str) -> bool:
    """Validate the processed data"""
    try:
        # Check minimum requirements
        if df.empty:
            logger.error(f"Processed data is empty for {ticker}")
            return False

        # Check required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Ticker']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns after processing for {ticker}: {missing_cols}")
            return False

        # Check for sufficient data
        if len(df) < 10:
            logger.error(f"Insufficient data after processing for {ticker}: {len(df)} rows")
            return False

        # Check that we have some valid price data
        price_cols = ['Open', 'High', 'Low', 'Close']
        if df[price_cols].isna().all().all():
            logger.error(f"No valid price data after processing for {ticker}")
            return False

        logger.info(f"Data validation passed for {ticker}: {len(df)} rows")
        return True

    except Exception as e:
        logger.error(f"Validation error for {ticker}: {str(e)}")
        return False