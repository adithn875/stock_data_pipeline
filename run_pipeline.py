import os
import logging
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import yfinance as yf

# Import your custom modules
from fetch_data import fetch_stock_data
from process_data import process_stock_data
from store_data import store_processed_data


class StockDataPipeline:
    def __init__(self, config_path: str = None):
        self.setup_logging()
        self.config = self._load_config(config_path)
        self.results = {"successful": [], "failed": []}

    def setup_logging(self):
        """Setup logging with UTF-8 encoding to handle Unicode characters"""
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)

        # Configure logging with UTF-8 encoding
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/pipeline.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )

        # Set console handler encoding to UTF-8
        for handler in logging.root.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream.name == '<stderr>':
                handler.stream.reconfigure(encoding='utf-8')

        self.logger = logging.getLogger(__name__)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration - simplified for this example"""
        return {
            "tickers": ["AAPL", "GOOGL", "TCS.NS", "RELIANCE.NS"],
            "period": "2y",
            "max_workers": 3
        }

    def _load_tickers(self) -> List[str]:
        """Load ticker symbols from configuration"""
        tickers = self.config.get("tickers", [])
        self.logger.info(f"Loaded {len(tickers)} tickers: {', '.join(tickers)}")
        return tickers

    def _process_single_ticker(self, ticker: str) -> Dict[str, Any]:
        """Process a single ticker through the entire pipeline"""
        try:
            self.logger.info(f"Starting pipeline for: {ticker}")

            # Step 1: Fetch raw data
            print(f"Fetching data for: {ticker}")
            raw_df = fetch_stock_data(ticker, period=self.config.get("period", "2y"))

            if raw_df is None or raw_df.empty:
                raise ValueError(f"No data fetched for {ticker}")

            # Step 2: Save raw data
            raw_file = f"data/raw/{ticker.replace('.', '_')}.csv"
            os.makedirs(os.path.dirname(raw_file), exist_ok=True)
            raw_df.to_csv(raw_file)
            print(f"Saved data to {raw_file}")

            # Step 3: Process data
            processed_df = process_stock_data(raw_df, ticker)  # Pass the DataFrame and ticker separately

            if processed_df is None or processed_df.empty:
                raise ValueError(f"No processed data for {ticker}")

            # Step 4: Store processed data
            store_processed_data(processed_df, ticker)

            return {
                "ticker": ticker,
                "status": "success",
                "rows": len(processed_df),
                "error": None
            }

        except Exception as e:
            error_msg = f"{ticker} failed: {e}"
            self.logger.error(error_msg)
            print(f"Error processing data for {ticker}: {e}")
            return {
                "ticker": ticker,
                "status": "failed",
                "rows": 0,
                "error": str(e)
            }

    def run_parallel(self, tickers: List[str]) -> None:
        """Run pipeline for multiple tickers in parallel"""
        max_workers = min(self.config.get("max_workers", 3), len(tickers))
        self.logger.info(f"Running parallel processing with {max_workers} workers...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_ticker = {
                executor.submit(self._process_single_ticker, ticker): ticker
                for ticker in tickers
            }

            # Process completed tasks
            for i, future in enumerate(as_completed(future_to_ticker), 1):
                ticker = future_to_ticker[future]
                result = future.result()

                if result["status"] == "success":
                    self.results["successful"].append(result)
                    self.logger.info(f"SUCCESS {ticker}: {result['rows']} rows processed")
                else:
                    self.results["failed"].append(result)
                    self.logger.error(f"FAILED {ticker}: {result['error']}")

                self.logger.info(f"Completed {i}/{len(tickers)}: {ticker}")

    def run_sequential(self, tickers: List[str]) -> None:
        """Run pipeline for multiple tickers sequentially"""
        self.logger.info("Running sequential processing...")

        for i, ticker in enumerate(tickers, 1):
            result = self._process_single_ticker(ticker)

            if result["status"] == "success":
                self.results["successful"].append(result)
            else:
                self.results["failed"].append(result)

            self.logger.info(f"Completed {i}/{len(tickers)}: {ticker}")

    def print_summary(self) -> None:
        """Print execution summary"""
        successful_count = len(self.results["successful"])
        failed_count = len(self.results["failed"])
        total_count = successful_count + failed_count

        print("\n" + "=" * 50)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 50)
        print(f"Successful: {successful_count}/{total_count}")
        print(f"Failed: {failed_count}/{total_count}")

        if self.results["successful"]:
            print(f"\nSuccessful tickers:")
            for result in self.results["successful"]:
                print(f"   • {result['ticker']}: {result['rows']} rows processed")

        if self.results["failed"]:
            print(f"\nFailed tickers:")
            for result in self.results["failed"]:
                print(f"   • {result['ticker']}: {result['error']}")

        print("=" * 50)


def main():
    """Main execution function"""
    print("Stock Data Pipeline Starting...")
    start_time = datetime.now()

    try:
        # Initialize pipeline
        pipeline = StockDataPipeline()

        # Load tickers
        tickers = pipeline._load_tickers()

        if not tickers:
            pipeline.logger.error("No tickers found in configuration")
            return

        # Create necessary directories
        os.makedirs("data/raw", exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)

        # Run pipeline (use parallel by default, change to sequential if needed)
        pipeline.run_parallel(tickers)
        # pipeline.run_sequential(tickers)  # Uncomment for sequential processing

        # Calculate execution time
        end_time = datetime.now()
        execution_time = end_time - start_time
        pipeline.logger.info(f"Total execution time: {execution_time}")

        # Print summary
        pipeline.print_summary()

    except Exception as e:
        logging.error(f"Pipeline failed with error: {e}")
        raise


if __name__ == "__main__":
    main()