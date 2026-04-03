import polars as pl
from pathlib import Path
import calendar
from datetime import date

DAILY_STOCK_FEATURES_PATH = Path(
    "data/pre-processor/prepared_data_2018-01-01_2023-12-31.parquet"
)
CUSIP_STOCKSYMBOL_MAP_PATH = Path("data/graphs/input_data/CUSIP.csv")
FTD_STOCK_MAP_PATH = Path("data/graphs/input_data/SEC_data/ftd")


def normalize_stock_symbol(lf, col_name):
    return lf.with_columns(
        pl.col(col_name)
        .str.strip_chars()
        .str.to_uppercase()
        .str.replace_all(r"[-./ ]", "")
        .alias(col_name)
    )


def ftd_cusip_stock_map_quarterly():
    """Reads FTD files, extracts the date from the filename, and snaps it to the quarter-end."""
    # Sort files for deterministic processing
    files = sorted(FTD_STOCK_MAP_PATH.glob("*.txt"))
    ftd_scans = []

    for f in files:
        name = f.stem
        try:
            # Extract year and month (e.g., 'cnsfails202301a' -> 2023, 01)
            date_str = "".join(filter(str.isdigit, name))[:6]
            if len(date_str) < 6:
                continue

            year = int(date_str[:4])
            month = int(date_str[4:6])

            # Map to SEC Quarter End (3, 6, 9, 12)
            q_month = ((month - 1) // 3 + 1) * 3
            last_day = calendar.monthrange(year, q_month)[1]
            report_date = date(year, q_month, last_day)

        except Exception:
            continue

        scan = (
            pl.scan_csv(
                f,
                separator="|",
                encoding="utf8-lossy",
                ignore_errors=True,
                truncate_ragged_lines=True,
                infer_schema_length=0,
            )
            .select(
                [
                    pl.col("CUSIP").alias("cusip"),
                    pl.col("SYMBOL").alias("Stock_symbol"),
                ]
            )
            .with_columns(pl.lit(report_date).alias("report_date"))
        )
        ftd_scans.append(scan)

    if not ftd_scans:
        # Fallback empty lazyframe if no txt files found
        return pl.LazyFrame(
            schema={"cusip": pl.Utf8, "Stock_symbol": pl.Utf8, "report_date": pl.Date}
        )

    return pl.concat(ftd_scans).drop_nulls().unique(subset=["report_date", "cusip"])


def load_stock_symbols_quarterly():
    # 1. Price Universe per Quarter
    stock_data_quarterly = (
        pl.scan_parquet(DAILY_STOCK_FEATURES_PATH)
        .with_columns(report_quarter=((((pl.col("Date").dt.month() - 1) // 3 + 1) * 3)))
        .with_columns(
            report_date=pl.date(
                pl.col("Date").dt.year(), pl.col("report_quarter"), 1
            ).dt.month_end()
        )
        .select(["report_date", "Stock_symbol"])
        .unique()
    )
    stock_data_quarterly = normalize_stock_symbol(stock_data_quarterly, "Stock_symbol")

    # 2. Get Quarterly FTD Map
    cusip_ftd_quarterly = normalize_stock_symbol(
        ftd_cusip_stock_map_quarterly(), "Stock_symbol"
    )

    # 3. Static CUSIP file (Fallback)
    static_cusip_map = normalize_stock_symbol(
        pl.scan_csv(CUSIP_STOCKSYMBOL_MAP_PATH).select(
            [
                pl.col("cusip"),
                pl.col("symbol").alias("Stock_symbol"),
            ]
        ),
        "Stock_symbol",
    )

    # 4. Temporal Join: Match Active Tickers with FTD data for that exact quarter
    temporal_bridge = stock_data_quarterly.join(
        cusip_ftd_quarterly, on=["Stock_symbol", "report_date"], how="left"
    )

    # 5. Fill Gaps: If a ticker wasn't in the FTD file, use the static CUSIP map
    temporal_bridge = (
        temporal_bridge.join(
            static_cusip_map, on="Stock_symbol", how="left", suffix="_static"
        )
        .with_columns(
            # Use FTD cusip first; if null, use the static cusip
            pl.coalesce(["cusip", "cusip_static"]).alias("cusip")
        )
        .drop("cusip_static")
        .drop_nulls("cusip")
        .unique()
    )

    return temporal_bridge


def create_stock_nodes(temporal_bridge_lf, sec_root_path):
    # 1. Submissions (With robust date snapping)
    submissions = (
        pl.scan_csv(
            f"{sec_root_path}/**/SUBMISSION.tsv",
            separator="\t",
            infer_schema_length=0,
            ignore_errors=True,
        )
        .select(
            [
                pl.col("ACCESSION_NUMBER"),
                # Use try_parse to handle multiple formats or invalid strings
                pl.col("PERIODOFREPORT")
                .str.to_date("%d-%b-%Y", strict=False)
                .alias("raw_date"),
            ]
        )
        .drop_nulls("raw_date")
        .with_columns(
            # SNAP to the nearest quarter end (3, 6, 9, 12)
            report_date=(
                pl.date(
                    pl.col("raw_date").dt.year(),
                    ((pl.col("raw_date").dt.month() - 1) // 3 + 1) * 3,
                    1,
                ).dt.month_end()
            )
        )
        .select(["ACCESSION_NUMBER", "report_date"])
    )

    # 2. InfoTable (Remains the same)
    holdings_scan = pl.scan_csv(
        f"{sec_root_path}/**/INFOTABLE.tsv",
        separator="\t",
        infer_schema_length=0,
        ignore_errors=True,
    ).select([pl.col("ACCESSION_NUMBER"), pl.col("CUSIP").alias("cusip")])

    # 3. The Join
    active_holdings_lf = holdings_scan.join(submissions, on="ACCESSION_NUMBER").join(
        temporal_bridge_lf, on=["cusip", "report_date"], how="inner"
    )

    stock_nodes_df = temporal_bridge_lf.select("Stock_symbol").unique().collect()

    return stock_nodes_df, active_holdings_lf


def analyze_quarterly_coverage(active_holdings_lf, temporal_bridge_lf):
    # SEC Stats
    sec_stats = active_holdings_lf.group_by("report_date").agg(
        [
            pl.col("Stock_symbol").n_unique().alias("unique_stocks_held"),
            pl.col("cusip").count().alias("total_records"),
        ]
    )

    # Price Stats (the denominator comes directly from our temporal bridge)
    price_stats = temporal_bridge_lf.group_by("report_date").agg(
        pl.col("Stock_symbol").n_unique().alias("total_price_symbols")
    )

    return (
        sec_stats.join(price_stats, on="report_date")
        .with_columns(
            ((pl.col("unique_stocks_held") / pl.col("total_price_symbols")) * 100)
            .round(2)
            .alias("coverage_pct")
        )
        .sort("report_date")
        .collect()
    )


if __name__ == "__main__":
    # 1. Generate the bridge that maps CUSIPs to Tickers PER QUARTER
    temporal_bridge_lf = load_stock_symbols_quarterly()

    sec_root = "data/graphs/input_data/SEC_data"

    # 2. Create the Nodes and Edges
    stock_nodes_df, active_holdings_lf = create_stock_nodes(
        temporal_bridge_lf, sec_root
    )

    print(f"Total Unique Stock Symbols in Universe: {len(stock_nodes_df)}")

    # 3. Analyze coverage (passing both lazy objects)
    coverage = analyze_quarterly_coverage(active_holdings_lf, temporal_bridge_lf)

    print("\n--- Quarterly Coverage Report (Corrected Alignment) ---")
    with pl.Config(tbl_rows=100):
        print(coverage)
