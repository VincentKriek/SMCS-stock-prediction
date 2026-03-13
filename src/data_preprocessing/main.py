from format_parquet import format_parquet
from async_llm_processor import generate_sentiment_scores
from aggregate_scores import aggregate_scores
from pathlib import Path
import os
import polars as pl
from dotenv import load_dotenv
import asyncio

load_dotenv()

LAMBDA_DECAY = 0.03


async def main():
    format_step(os.environ["MIN_DATE"], os.environ["MAX_DATE"])
    await generate_sentiment_scores()


def format_step(min_date, max_date):
    input_file = Path(f"data/loader/news_loaded_{min_date}_{max_date}.parquet")
    output_file = Path(
        f"data/pre-processor/news_formatted_{min_date}_{max_date}.parquet"
    )

    lf = pl.scan_parquet(input_file)
    lf = format_parquet(lf)
    lf.sink_parquet(output_file)


def aggregate_step(min_date, max_date):
    input_news_file = Path(
        f"data/pre-processor/news_scored_{min_date}_{max_date}.parquet"
    )
    input_stock_file = Path(f"data/loader/prices_loaded_{min_date}_{max_date}.parquet")
    output_file = Path(
        f"data/pre-processor/prepared_date_{min_date}_{max_date}.parquet"
    )

    news_lf = pl.scan_parquet(input_news_file)
    stock_lf = pl.scan_parquet(input_stock_file)
    final_lf = aggregate_scores(news_lf, stock_lf, LAMBDA_DECAY)
    final_lf.sink_parquet(output_file)


if __name__ == "__main__":
    asyncio.run(main())
