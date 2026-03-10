from huggingface_hub import hf_hub_download
import polars as pl
import zipfile
from pathlib import Path

EXPECTED_COLS = ["date", "open", "high", "low", "close", "adj close", "volume"]

def read_and_reorder_csv(csv_file: Path):
    # Read CSV normally
    df = pl.read_csv(csv_file, infer_schema_length=10000)

    # Find which columns exist in this file
    cols_in_file = df.columns
    cols_to_select = [c for c in EXPECTED_COLS if c in cols_in_file]

    # Reorder the columns, fix the types
    df = df.select(cols_to_select)
    df = df.with_columns([
        pl.col("date").cast(pl.Utf8),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("adj close").cast(pl.Float64),
        pl.col("volume").cast(pl.Int64),
    ])

    return df

def load_and_reorder_csvs(csv_files) -> pl.LazyFrame:
    """
    Scans all CSVs, reorders columns according to a set schema`,
    and returns a single LazyFrame.
    """
    dfs = [read_and_reorder_csv(f) for f in csv_files]
    df_all = pl.concat(dfs).rename({"date": "Date"})
    return pl.LazyFrame(df_all)

def load_hf_lazyframe(repo_id, subfolder, filename, min_date, max_date):
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
        repo_type="dataset",
    )

    return (
        pl.scan_csv(
            path,
            ignore_errors=True,
            infer_schema_length=10000,
        )
        .with_columns(
            pl.col("Date")
            .str.replace(" UTC", "")
            .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
        )
        .filter(pl.col("Date").is_between(min_date, max_date))
    )

def load_hf_prices_lazyframe(repo_id, subfolder, filename, min_date, max_date) -> pl.LazyFrame:

    # Download zip from Hugging Face
    zip_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
        repo_type="dataset",
    )

    # Extract zip next to the downloaded file
    extract_dir = Path(zip_path).with_suffix("")
    if not extract_dir.exists():
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)

    # Find all CSV files
    csv_files = [
        p for p in extract_dir.rglob("*.csv")
        if not p.name.startswith("._") and "__MACOSX" not in str(p)
    ]

    lf = load_and_reorder_csvs(csv_files) # re-order columns in a consistent schema

    # Process Date column and filter
    return (
        lf.with_columns(
            pl.col("Date")
            .str.replace(" UTC", "")
            .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
        )
        .filter(pl.col("Date").is_between(min_date, max_date))
    )