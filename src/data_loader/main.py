from dotenv import load_dotenv
import os
from load_huggingface import load_hf_lazyframe
from clean_data import remove_unneccessary_columns
from news_scraper import add_article_column_stream
from summarize import add_summary_column
import polars as pl
from pathlib import Path
from datetime import datetime


load_dotenv()
CHUNK_SIZE = 7  # chunk size in days


def load_data():
    repo_id = os.environ["HF_REPO_ID"]
    subfolder = os.environ["HF_SUBFOLDER"]
    min_date = datetime.strptime(os.environ["MIN_DATE"], "%Y-%m-%d")
    max_date = datetime.strptime(os.environ["MAX_DATE"], "%Y-%m-%d")

    # load data into lazyframe
    all_external_lf = load_hf_lazyframe(
        repo_id, subfolder, "All_external.csv", min_date, max_date
    )
    nasdaq_lf = load_hf_lazyframe(
        repo_id, subfolder, "nasdaq_exteral_data.csv", min_date, max_date
    )

    # processing of external news
    all_external_lf = load_external_data(all_external_lf)

    # processing of nasdaq news
    nasdaq_lf = load_nasdaq_data(nasdaq_lf)

    # concat lazyframes
    news_lf = pl.concat([all_external_lf, nasdaq_lf])

    news_lf.sink_parquet(Path("data/news_data_loader.parquet"), compression="zstd")

    return news_lf


def load_external_data(lf):
    return (
        lf.pipe(add_article_column_stream)
        .pipe(add_summary_column)
        .pipe(remove_unneccessary_columns)
    )


def load_nasdaq_data(lf):
    return lf.pipe(add_summary_column).pipe(remove_unneccessary_columns)


def combine_chunks():
    pass


if __name__ == "__main__":
    load_data()
