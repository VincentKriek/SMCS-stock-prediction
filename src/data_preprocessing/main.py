from format_parquet import format_parquet
from async_llm_processor import generate_sentiment_scores
from pathlib import Path
import os
import polars as pl
from dotenv import load_dotenv
import asyncio

load_dotenv()


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


if __name__ == "__main__":
    asyncio.run(main())
