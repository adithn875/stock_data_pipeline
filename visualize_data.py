# src/visualize_data.py - Working Complete Version

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import warnings
import numpy as np

warnings.filterwarnings('ignore')


def visualize_stock_data(ticker, save_plots=False):
    """
    Create comprehensive visualizations for stock data
    """
    try:
        # Handle different ticker input formats
        if ticker.endswith('.csv'):
            # If ticker is a filename, extract the actual ticker
            base_name = ticker.replace('.csv', '')
            ticker_clean = base_name.replace('_csv', '').replace('_processed', '')
        else:
            ticker_clean = ticker

        # Try different file path patterns
        possible_files = [
            f"data/processed/{ticker_clean}_processed.csv",
            f"data/processed/{ticker.replace('.csv', '')}_processed.csv",
            f"data/processed/{ticker.replace('.csv', '').replace('_csv', '')}_processed.csv"
        ]

        processed_file = None
        for file_path in possible_files:
            if os.path.exists(file_path):
                processed_file = file_path
                break

        # If no processed file found, search in directory
        if not processed_file:
            processed_dir = "data/processed"
            if os.path.exists(processed_dir):
                available_files = [f for f in os.listdir(processed_dir) if f.endswith('_processed.csv')]
                print(f"Available processed files: {available_files}")

                # Try to find a matching file
                for available_file in available_files:
                    if ticker_clean.upper().replace('.', '_') in available_file.upper():
                        processed_file = f"{processed_dir}/{available_file}"
                        print(f"Found matching file: {processed_file}")
                        break

        if not processed_file or not os.path.exists(processed_file):
            raise FileNotFoundError(f"Processed data file not found for ticker: {ticker_clean}")

        # Read the processed data
        print(f"📊 Loading data from: {processed_file}")
        df = pd.read_csv(processed_file)

        # Ensure Date column is datetime and set as index
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        elif df.index.name != 'Date':
            # Try to convert index to datetime if it's not already
            try:
                df.index = pd.to_datetime(df.index)
            except:
                pass

        # Create the plots directory if it doesn't exist
        if save_plots:
            plots_dir = "plots"
            os.makedirs(plots_dir, exist_ok=True)

        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")

        # Create a figure with subplots
        fig = plt.figure(figsize=(20, 16))

        # Extract ticker name for titles
        ticker_name = ticker_clean.replace('_processed', '').upper()

        # 1. Price Chart
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(df.index, df['Close'], linewidth=2, label='Close Price', color='#2E86AB')
        ax1.fill_between(df.index, df['Close'], alpha=0.3, color='#2E86AB')
        ax1.set_title(f'{ticker_name} - Stock Price Over Time', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)

        # 2. Volume Chart
        ax2 = plt.subplot(3, 2, 2)
        ax2.bar(df.index, df['Volume'], alpha=0.7, color='#A23B72', width=1)
        ax2.set_title(f'{ticker_name} - Trading Volume', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)

        # 3. OHLC Chart (Candlestick-style)
        ax3 = plt.subplot(3, 2, 3)
        # Limit data points for better visibility
        sample_data = df.iloc[-100:] if len(df) > 100 else df

        for i, (date, row) in enumerate(sample_data.iterrows()):
            color = '#00A86B' if row['Close'] >= row['Open'] else '#FF6B6B'
            ax3.plot([date, date], [row['Low'], row['High']], color=color, linewidth=1)
            ax3.plot([date, date], [row['Open'], row['Close']], color=color, linewidth=3)

        ax3.set_title(f'{ticker_name} - OHLC Chart (Last 100 days)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Price ($)', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)

        # 4. Moving Averages (if available)
        ax4 = plt.subplot(3, 2, 4)
        ax4.plot(df.index, df['Close'], label='Close Price', linewidth=2, color='#2E86AB')

        # Check if moving averages exist in the data
        ma_columns = [col for col in df.columns if any(x in col.upper() for x in ['MA', 'SMA', 'EMA'])]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

        for i, ma_col in enumerate(ma_columns[:5]):  # Limit to 5 MAs
            if i < len(colors):
                color = colors[i]
                ax4.plot(df.index, df[ma_col], label=ma_col, linewidth=1.5, color=color)

        ax4.set_title(f'{ticker_name} - Price with Moving Averages', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Price ($)', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)

        # 5. Technical Indicators
        ax5 = plt.subplot(3, 2, 5)

        # Check for RSI
        if 'RSI' in df.columns:
            ax5.plot(df.index, df['RSI'], label='RSI', color='#E17055', linewidth=2)
            ax5.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
            ax5.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
            ax5.set_ylim(0, 100)
            ax5.set_title(f'{ticker_name} - RSI Indicator', fontsize=14, fontweight='bold')
            ax5.set_ylabel('RSI', fontsize=12)
            ax5.legend()
        else:
            # If no RSI, show daily returns
            if 'Daily_Return' in df.columns:
                ax5.plot(df.index, df['Daily_Return'], label='Daily Returns', color='#6C5CE7', linewidth=1)
                ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                ax5.set_title(f'{ticker_name} - Daily Returns', fontsize=14, fontweight='bold')
                ax5.set_ylabel('Daily Return (%)', fontsize=12)
                ax5.legend()
            else:
                ax5.text(0.5, 0.5, 'No technical indicators available',
                         transform=ax5.transAxes, ha='center', va='center', fontsize=12)
                ax5.set_title(f'{ticker_name} - Technical Indicators', fontsize=14, fontweight='bold')

        ax5.grid(True, alpha=0.3)
        ax5.tick_params(axis='x', rotation=45)

        # 6. Price Distribution
        ax6 = plt.subplot(3, 2, 6)
        ax6.hist(df['Close'], bins=50, alpha=0.7, color='#74B9FF', edgecolor='black')
        ax6.set_title(f'{ticker_name} - Price Distribution', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Price ($)', fontsize=12)
        ax6.set_ylabel('Frequency', fontsize=12)
        ax6.grid(True, alpha=0.3)

        # Adjust layout
        plt.tight_layout(pad=3.0)

        # Save the plot if requested
        if save_plots:
            plot_filename = f"plots/{ticker_name}_comprehensive_analysis.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"💾 Plot saved as: {plot_filename}")

        # Show the plot
        plt.show()

        # Print summary statistics
        print(f"\n📊 Summary Statistics for {ticker_name}:")
        print("=" * 50)
        print(f"Data Period: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
        print(f"Total Trading Days: {len(df)}")
        print(f"Current Price: ${df['Close'].iloc[-1]:.2f}")
        print(f"Highest Price: ${df['High'].max():.2f}")
        print(f"Lowest Price: ${df['Low'].min():.2f}")
        print(f"Average Volume: {df['Volume'].mean():,.0f}")

        if 'Daily_Return' in df.columns:
            print(f"Average Daily Return: {df['Daily_Return'].mean():.2f}%")
            print(f"Volatility (Std Dev): {df['Daily_Return'].std():.2f}%")

        print("✅ Visualization completed successfully!")

    except FileNotFoundError as e:
        print(f"❌ File Error: {e}")
        print("💡 Make sure the processed data file exists in the data/processed/ directory")
        print("💡 Run options 1-4 first to process your data")
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()


def create_visualizations(ticker, df=None, save_plots=False):
    """
    Alternative function name for compatibility with existing imports.
    """
    if df is not None:
        # If dataframe is provided, save it temporarily and use visualize_stock_data
        temp_file = f"data/processed/{ticker}_temp_processed.csv"
        os.makedirs("data/processed", exist_ok=True)
        df.to_csv(temp_file)

        try:
            result = visualize_stock_data(ticker, save_plots=save_plots)
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
            return result
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_file):
                os.remove(temp_file)
            raise e
    else:
        # If no dataframe provided, use the ticker directly
        return visualize_stock_data(ticker, save_plots=save_plots)


def create_correlation_heatmap(ticker, save_plots=False):
    """
    Create a correlation heatmap for technical indicators
    """
    try:
        # Load processed data
        processed_file = f"data/processed/{ticker}_processed.csv"
        if not os.path.exists(processed_file):
            raise FileNotFoundError(f"Processed data file not found: {processed_file}")

        df = pd.read_csv(processed_file, index_col=0, parse_dates=True)

        # Select numeric columns for correlation
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) < 2:
            print("❌ Not enough numeric columns for correlation analysis")
            return

        # Create correlation matrix
        corr_matrix = df[numeric_cols].corr()

        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title(f'{ticker.upper()} - Technical Indicators Correlation Matrix',
                  fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_plots:
            plt.savefig(f"plots/{ticker}_correlation_heatmap.png", dpi=300, bbox_inches='tight')
            print(f"💾 Correlation heatmap saved")

        plt.show()

    except Exception as e:
        print(f"❌ Error creating correlation heatmap: {e}")


def create_candlestick_chart(ticker, days=100, save_plots=False):
    """
    Create a detailed candlestick chart
    """
    try:
        # Load processed data
        processed_file = f"data/processed/{ticker}_processed.csv"
        if not os.path.exists(processed_file):
            raise FileNotFoundError(f"Processed data file not found: {processed_file}")

        df = pd.read_csv(processed_file, index_col=0, parse_dates=True)

        # Get last N days
        recent_data = df.tail(days)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10),
                                       gridspec_kw={'height_ratios': [3, 1]})

        # Candlestick chart
        for i, (date, row) in enumerate(recent_data.iterrows()):
            color = '#00A86B' if row['Close'] >= row['Open'] else '#FF6B6B'

            # High-low line
            ax1.plot([date, date], [row['Low'], row['High']],
                     color=color, linewidth=1, alpha=0.8)

            # Open-close body
            body_height = abs(row['Close'] - row['Open'])
            body_bottom = min(row['Open'], row['Close'])

            ax1.bar(date, body_height, bottom=body_bottom,
                    color=color, alpha=0.8, width=0.8)

        ax1.set_title(f'{ticker.upper()} - Candlestick Chart (Last {days} days)',
                      fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Volume chart
        colors = ['#00A86B' if close >= open_price else '#FF6B6B'
                  for close, open_price in zip(recent_data['Close'], recent_data['Open'])]

        ax2.bar(recent_data.index, recent_data['Volume'],
                color=colors, alpha=0.7, width=0.8)
        ax2.set_title('Volume', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Volume', fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plots:
            plt.savefig(f"plots/{ticker}_candlestick_chart.png", dpi=300, bbox_inches='tight')
            print(f"💾 Candlestick chart saved")

        plt.show()

    except Exception as e:
        print(f"❌ Error creating candlestick chart: {e}")


# Simple fallback function for basic plotting
def simple_plot(ticker):
    """
    Simple plotting function as a fallback
    """
    try:
        processed_file = f"data/processed/{ticker}_processed.csv"
        if not os.path.exists(processed_file):
            print(f"❌ File not found: {processed_file}")
            return

        df = pd.read_csv(processed_file, index_col=0, parse_dates=True)

        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['Close'], linewidth=2, label='Close Price')
        plt.title(f'{ticker.upper()} - Stock Price')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        print(f"✅ Simple plot created for {ticker}")

    except Exception as e:
        print(f"❌ Error creating simple plot: {e}")


if __name__ == "__main__":
    # Test the visualization functions
    print("🧪 Testing visualization functions...")

    # Test with sample data if available
    test_files = ["data/processed/AAPL_processed.csv", "data/processed/GOOGL_processed.csv"]

    for test_file in test_files:
        if os.path.exists(test_file):
            ticker = os.path.basename(test_file).replace('_processed.csv', '')
            print(f"Testing with {ticker}...")
            simple_plot(ticker)
            break
    else:
        print("No test files found. Create some processed data first.")