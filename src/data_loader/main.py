from dotenv import load_dotenv
import os
from load_huggingface import load_hf_lazyframe
from clean_data import remove_unneccessary_columns
from news_scraper import add_article_column_stream
from summarize import add_summary_column
import polars as pl
from pathlib import Path


load_dotenv()


def load_data():
    repo_id = os.environ["HF_REPO_ID"]
    subfolder = os.environ["HF_SUBFOLDER"]

    # load data into lazyframe
    all_external_lf = load_hf_lazyframe(repo_id, subfolder, "All_external.csv")
    nasdaq_lf = load_hf_lazyframe(repo_id, subfolder, "nasdaq_exteral_data.csv")

    # processing of external news
    all_external_lf = (
        all_external_lf.pipe(add_article_column_stream)
        .pipe(add_summary_column)
        .pipe(remove_unneccessary_columns)
    )

    # processing of nasdaq news
    nasdaq_lf = nasdaq_lf.pipe(add_summary_column).pipe(remove_unneccessary_columns)

    # concat lazyframes
    news_lf = pl.concat([all_external_lf, nasdaq_lf])

    print(news_lf.collect())
    news_lf.sink_parquet(Path("data/news_data_loader.parquet"), compression="zstd")

    return news_lf


if __name__ == "__main__":
    load_data()
