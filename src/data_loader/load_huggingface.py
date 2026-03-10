from huggingface_hub import hf_hub_download
import polars as pl
import zipfile
from pathlib import Path

EXPECTED_COLS = ["date", "open", "high", "low", "close", "adj close", "volume"]

def read_and_reorder_csv(csv_file: Path):
    # Read CSV normally (let Polars infer types)
    df = pl.read_csv(csv_file, infer_schema_length=10000)
    if csv_file.name in ["A.csv", "AA.csv"]:
        print(f"Preview of {csv_file.name}:")
        print(df.head(5))  # collects first 5 rows and prints actual data

    # Find which columns exist in this file
    cols_in_file = df.columns
    cols_to_select = [c for c in EXPECTED_COLS if c in cols_in_file]
    if csv_file.name in ["A.csv", "AA.csv"]:
        print(cols_to_select)

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
    
    if csv_file.name in ["A.csv", "AA.csv"]:
        print(f"Postview of {csv_file.name}:")
        print(df.head(5))  # collects first 5 rows and prints actual data

    return df

def load_and_reorder_csvs(csv_files) -> pl.LazyFrame:
    """
    Scans all CSVs, reorders columns according to a set schema`,
    and returns a single LazyFrame.
    """
    dfs = [read_and_reorder_csv(f) for f in csv_files]
    # for i, df in enumerate(dfs):
    #     print(df.collect_schema())
    df_all = pl.concat(dfs).rename({"date": "Date"})
    # lfs = []

    # for i, csv_file in enumerate(csv_files):
    #     lf = pl.scan_csv(csv_file, schema=PRICE_COL_SCHEMA)
    #     # if i  < 2:
    #     #     print(csv_file)
    #     #     print(lf.columns)
    #     #     print(lf.head(5).collect())
    #     # lf = lf.select(expected_columns)
    #     lfs.append(lf)
     
    # # print("-"*50)
    # lf:pl.LazyFrame = pl.concat(lfs)
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
    # print(csv_files[0])

    lf = load_and_reorder_csvs(csv_files) # re-order columns in a consistent schema
    # print(lf.head())

    # Scan all CSVs lazily
    # lf = pl.scan_csv(
    #     [str(p) for p in csv_files],
    #     ignore_errors=True,
    #     infer_schema_length=10000,
    # )

    # Process Date column and filter
    return (
        lf.with_columns(
            pl.col("Date")
            .str.replace(" UTC", "")
            .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
        )
        .filter(pl.col("Date").is_between(min_date, max_date))
    )