from dotenv import load_dotenv
import os
from load_huggingface import load_hf_lazyframe, load_hf_prices_lazyframe
from clean_data import remove_unneccessary_columns
from news_scraper import add_article_column_stream
from summarize import add_summary_column
import polars as pl
from pathlib import Path
from datetime import datetime, timedelta
from math import ceil
from tqdm import tqdm

load_dotenv()
BATCH_DAYS = 30  # batch size in days


def load_data():
    # repo_id = os.environ["HF_REPO_ID"]
    # subfolder = os.environ["HF_SUBFOLDER"]
    # min_date = datetime.strptime(os.environ["MIN_DATE"], "%Y-%m-%d")
    # max_date = datetime.strptime(os.environ["MAX_DATE"], "%Y-%m-%d")

    repo_id = "Zihan1004/FNSPID"
    subfolder = "Stock_price"
    min_date = datetime.strptime("2019-01-01", "%Y-%m-%d")
    max_date = datetime.strptime("2020-01-01", "%Y-%m-%d")

    # load stock price data
    prices_lf = load_hf_prices_lazyframe(
        repo_id, subfolder, "full_history.zip", min_date, max_date
    )
    
    # p = Path("C:/Users/micha/.cache/huggingface/hub/datasets--Zihan1004--FNSPID/snapshots/bf9189c41527198897d1af3e17b1a0095279fc45/Stock_price/full_history/full_history")
    # test_file = p / "zyxi.csv"
    # test_file2 = p / "zumz.csv"
    
    prices_lf = load_price_data(prices_lf, min_date, max_date)

    # load data into lazyframe
    # all_external_lf = load_hf_lazyframe(
    #     repo_id, subfolder, "All_external.csv", min_date, max_date
    # )
    # nasdaq_lf = load_hf_lazyframe(
    #     repo_id, subfolder, "nasdaq_exteral_data.csv", min_date, max_date
    # )

    # # processing of external news
    # all_external_lf = load_external_data(all_external_lf, min_date, max_date)

    # # processing of nasdaq news
    # nasdaq_lf = load_nasdaq_data(nasdaq_lf, min_date, max_date)

# def check_column_order(folder):
#     reference = None

#     for path in Path(folder).glob("*.csv"):
#         with open(path) as f:
#             cols = tuple(f.readline().strip().split(","))

#         if reference is None:
#             reference = cols
#             continue

#         if cols != reference:
#             print(f"\n{path} has different column order")
#             print("expected:", reference)
#             print("got     :", cols)

# def enforce_schema(lf: pl.DataFrame):
#     # col_order = ["date", "volume", "open", "high", "low", "close", "adj close"]
#     ordered_lf = lf.select("date", "volume", "open", "high", "low", "close", "adj close")
#     print(ordered_lf.collect_schema())
#     return ordered_lf

def load_price_data(lf, start_date, end_date):
    total_batches = ceil((end_date - start_date).days / BATCH_DAYS)
    existing_ranges = get_existing_ranges(Path("data/loader/batches/price"))

    batch_lf, current_date, next_current_date, batch_start, batch_end = create_batch(
        lf, start_date, end_date, existing_ranges
    )

    with tqdm(total=total_batches, desc="Processing price batches") as pbar:
        while current_date < end_date:
            if batch_lf is not None:
                out_path = Path(
                    f"data/loader/batches/price/{batch_start.date()}_{batch_end.date()}.parquet"
                )
                out_path.parent.mkdir(parents=True, exist_ok=True)
                batch_lf.sink_parquet(out_path, compression="zstd")


            batch_lf, current_date, next_current_date, batch_start, batch_end = (
                create_batch(lf, next_current_date, end_date, existing_ranges)
            )

            pbar.update(1)

def load_external_data(lf, start_date, end_date):
    total_batches = ceil((end_date - start_date).days / BATCH_DAYS)
    existing_ranges = get_existing_ranges(Path("data/loader/batches/external"))

    batch_lf, current_date, next_current_date, batch_start, batch_end = create_batch(
        lf, start_date, end_date, existing_ranges
    )

    with tqdm(total=total_batches, desc="Processing external batches") as pbar:
        while current_date < end_date:
            if batch_lf is not None:
                batch_lf = (
                    batch_lf.pipe(add_article_column_stream)
                    .pipe(add_summary_column)
                    .pipe(remove_unneccessary_columns)
                )

                out_path = Path(
                    f"data/loader/batches/external/{batch_start.date()}_{batch_end.date()}.parquet"
                )
                batch_lf.sink_parquet(out_path, compression="zstd")

            batch_lf, current_date, next_current_date, batch_start, batch_end = (
                create_batch(lf, next_current_date, end_date, existing_ranges)
            )

            pbar.update(1)


def load_nasdaq_data(lf, start_date, end_date):
    total_batches = ceil((end_date - start_date).days / BATCH_DAYS)
    existing_ranges = get_existing_ranges(Path("data/loader/batches/nasdaq"))

    batch_lf, current_date, next_current_date, batch_start, batch_end = create_batch(
        lf, start_date, end_date, existing_ranges
    )

    with tqdm(total=total_batches, desc="Processing nasdaq batches") as pbar:
        while current_date < end_date:
            if batch_lf is not None:
                batch_lf = batch_lf.pipe(add_summary_column).pipe(
                    remove_unneccessary_columns
                )

                out_path = Path(
                    f"data/loader/batches/nasdaq/{batch_start.date()}_{batch_end.date()}.parquet"
                )
                batch_lf.sink_parquet(out_path, compression="zstd")

            batch_lf, current_date, next_current_date, batch_start, batch_end = (
                create_batch(lf, next_current_date, end_date, existing_ranges)
            )

            pbar.update(1)


def dates_in_range(min_date, max_date):
    return [min_date + timedelta(days=i) for i in range((max_date - min_date).days + 1)]


def create_batch(lf, current_date, end_date, existing_date_ranges):
    next_current_date = current_date + timedelta(days=BATCH_DAYS)

    if current_date > end_date:
        return None, current_date, next_current_date, None, None

    range_dates = dates_in_range(current_date, next_current_date)
    unprocessed_dates = [
        d for d in range_dates if not is_processed(d, existing_date_ranges)
    ]
    if len(unprocessed_dates) > 0:
        batch_start, batch_end = min(unprocessed_dates), max(unprocessed_dates)

        batch_lf = lf.filter(pl.col("Date").is_between(batch_start, batch_end))
    else:
        batch_lf = batch_start = batch_end = None

    next_current_date += timedelta(
        days=1
    )  # adjust for closed in_between set to prevent duplicates

    return batch_lf, current_date, next_current_date, batch_start, batch_end


def get_existing_ranges(folder: Path):
    """
    Scan parquet files and extract date ranges.
    Expects filenames like 'external_YYYY-MM-DD_YYYY-MM-DD.parquet'
    """
    ranges = []
    for f in folder.glob("*.parquet"):
        parts = f.stem.split("_")
        if len(parts) == 2:
            start, end = datetime.fromisoformat(parts[0]), datetime.fromisoformat(
                parts[1]
            )
            ranges.append((start, end))
    return ranges


def is_processed(date, existing_ranges):
    return any(start <= date <= end for start, end in existing_ranges)


def combine_chunks():
    pass


if __name__ == "__main__":
    load_data()
