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


def create_active_holdings(temporal_bridge_lf, sec_root_path):
    """
    Combines Submissions and InfoTables, then joins with the temporal bridge
    to produce a LazyFrame of all valid bank-stock holdings.
    """
    # 1. Submissions (Standardizing the reporting dates)
    submissions = (
        pl.scan_csv(
            f"{sec_root_path}/**/SUBMISSION.tsv",
            separator="\t",
            infer_schema_length=0,
            ignore_errors=True,
            truncate_ragged_lines=True,
        )
        .select(
            [
                pl.col("ACCESSION_NUMBER"),
                pl.col("CIK").alias("bank_id"),  # Need this for the bank nodes later!
                pl.col("PERIODOFREPORT")
                .str.to_date("%d-%b-%Y", strict=False)
                .alias("raw_date"),
            ]
        )
        .drop_nulls("raw_date")
        .with_columns(
            report_date=(
                pl.date(
                    pl.col("raw_date").dt.year(),
                    ((pl.col("raw_date").dt.month() - 1) // 3 + 1) * 3,
                    1,
                ).dt.month_end()
            )
        )
        .select(["ACCESSION_NUMBER", "bank_id", "report_date"])
    )

    # 2. InfoTable (Grabbing the numeric 'Weights')
    holdings_scan = pl.scan_csv(
        f"{sec_root_path}/**/INFOTABLE.tsv",
        separator="\t",
        infer_schema_length=0,
        ignore_errors=True,
        truncate_ragged_lines=True,
    ).select(
        [
            pl.col("ACCESSION_NUMBER"),
            pl.col("CUSIP").alias("cusip"),
            pl.col("VALUE").cast(pl.Float64, strict=False).fill_null(0),
            pl.col("SSHPRNAMT").cast(pl.Float64, strict=False).fill_null(0),
            pl.col("VOTING_AUTH_SOLE").cast(pl.Float64, strict=False).fill_null(0),
            pl.col("VOTING_AUTH_SHARED").cast(pl.Float64, strict=False).fill_null(0),
            pl.col("VOTING_AUTH_NONE").cast(pl.Float64, strict=False).fill_null(0),
        ]
    )

    # 3. The Triple Join: InfoTable + Submissions + Temporal Bridge
    # This ensures we only keep holdings for stocks we have price data for.
    active_holdings_lf = holdings_scan.join(submissions, on="ACCESSION_NUMBER").join(
        temporal_bridge_lf, on=["cusip", "report_date"], how="inner"
    )

    return active_holdings_lf


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


def extract_metadata(sec_root_path):
    # We need SUBMISSION to link Accession Numbers to CIKs (bank_id)
    subs = pl.scan_csv(
        f"{sec_root_path}/**/SUBMISSION.tsv",
        separator="\t",
        infer_schema_length=0,
        ignore_errors=True,
    ).select([pl.col("ACCESSION_NUMBER"), pl.col("CIK").alias("bank_id")])

    bank_meta = (
        pl.scan_csv(
            f"{sec_root_path}/**/COVERPAGE.tsv",
            separator="\t",
            infer_schema_length=0,
            ignore_errors=True,
        )
        .join(subs, on="ACCESSION_NUMBER")  # Now we have bank_id!
        .select(
            [
                "bank_id",
                pl.col("FILINGMANAGER_NAME").alias("bank_name"),
                pl.col("FILINGMANAGER_CITY").alias("city"),
                pl.col("FILINGMANAGER_STATEORCOUNTRY").alias("state"),
            ]
        )
    )

    stock_meta = pl.scan_csv(
        f"{sec_root_path}/**/INFOTABLE.tsv",
        separator="\t",
        infer_schema_length=0,
        ignore_errors=True,
    ).select(
        [
            pl.col("CUSIP").alias("cusip"),
            pl.col("NAMEOFISSUER").alias("stock_name_raw"),
            pl.col("TITLEOFCLASS").alias("title_of_class"),
        ]
    )

    return bank_meta, stock_meta


def build_gnn_tables(active_holdings_lf, bank_meta_lf, stock_meta_lf):
    print("Aggregating Edge Weights and Node Features with Temporal Context...")

    # 0. Prep Temporal Data
    active_holdings_lf = active_holdings_lf.with_columns(
        [
            pl.col("report_date").dt.year().alias("year"),
            ((pl.col("report_date").dt.month() - 1) // 3 + 1).alias("quarter"),
        ]
    )

    # --- 1. EDGES (Bank-Stock) ---
    # We need sole, shared, AND none for the consumer script
    edges_bs = (
        active_holdings_lf.group_by(
            ["bank_id", "Stock_symbol", "year", "quarter", "report_date"]
        )
        .agg(
            [
                pl.sum("VALUE").alias("total_value"),
                pl.sum("SSHPRNAMT").alias("total_shares"),
                pl.sum("VOTING_AUTH_SOLE").alias("voting_sole"),
                pl.sum("VOTING_AUTH_SHARED").alias("voting_shared"),
                pl.sum("VOTING_AUTH_NONE").alias("voting_none"),
            ]
        )
        .rename({"Stock_symbol": "stock_id"})
    )

    # --- 2. BANK NODES ---
    # Calculate AUM, Count, Avg Position, and Persistence
    bank_nodes = (
        edges_bs.group_by(["bank_id", "year", "quarter", "report_date"])
        .agg(
            [
                pl.sum("total_value").alias("total_aum_value"),
                pl.col("stock_id").n_unique().alias("num_stocks_held"),
            ]
        )
        .with_columns(
            (pl.col("total_aum_value") / pl.col("num_stocks_held")).alias(
                "avg_position_size"
            )
        )
        # Calculate 'num_quarters_active' up to this point
        .with_columns(
            num_quarters_active=pl.col("report_date").cum_count().over("bank_id")
        )
        .join(bank_meta_lf.group_by("bank_id").first(), on="bank_id", how="left")
    )

    # --- 3. STOCK NODES ---
    # Calculate Holders, Value, Total Shares, and Persistence
    stock_nodes = (
        edges_bs.group_by(["stock_id", "year", "quarter", "report_date"])
        .agg(
            [
                pl.col("bank_id").n_unique().alias("num_holders"),
                pl.sum("total_value").alias("total_institutional_value"),
                pl.sum("total_shares").alias("total_institutional_shares"),
            ]
        )
        # Calculate 'num_quarters_held' up to this point
        .with_columns(
            num_quarters_held=pl.col("report_date").cum_count().over("stock_id")
        )
        .join(
            stock_meta_lf.group_by("cusip").first(),
            left_on="stock_id",
            right_on="cusip",
            how="left",
        )
    )

    return edges_bs, bank_nodes, stock_nodes


def build_stock_stock_edges(edges_bs_lf, max_portfolio_size=100, min_co_hold=2):
    # Change "Stock_symbol" to "stock_id" everywhere in this function
    portfolio_counts = (
        edges_bs_lf.group_by(["bank_id", "report_date"])
        .agg(pl.count("stock_id").alias("portfolio_size"))  # changed
        .filter(pl.col("portfolio_size") <= max_portfolio_size)
    )

    base = edges_bs_lf.join(
        portfolio_counts, on=["bank_id", "report_date"], how="inner"
    ).select(
        ["bank_id", "stock_id", "report_date"]
    )  # changed

    ss_edges = (
        base.join(base, on=["bank_id", "report_date"], how="inner", suffix="_right")
        .filter(pl.col("stock_id") < pl.col("stock_id_right"))  # changed
        .group_by(["stock_id", "stock_id_right", "report_date"])  # changed
        .agg(pl.count("bank_id").alias("co_holder_count"))
        .filter(pl.col("co_holder_count") >= min_co_hold)
        .rename({"stock_id": "stock_id_1", "stock_id_right": "stock_id_2"})  # changed
        .with_columns(
            pl.lit("co_held_by_institution").alias("edge_type"),
            # Add year/quarter for the consumer's filter
            pl.col("report_date").dt.year().alias("year"),
            ((pl.col("report_date").dt.month() - 1) // 3 + 1).alias("quarter"),
        )
    )
    return ss_edges


if __name__ == "__main__":
    sec_root = "data/graphs/input_data/SEC_data"
    output_path = Path("data/graphs")
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Generate the bridge (Lazy)
    temporal_bridge_lf = load_stock_symbols_quarterly()

    # 2. Create base holdings (Lazy - now includes VALUE/SHARES)
    # Note: Modified create_stock_nodes to return the LazyFrame
    active_holdings_lf = create_active_holdings(temporal_bridge_lf, sec_root)

    # 3. RUN COVERAGE TEST (This collects data to show you the report)
    print("\n--- Running Quarterly Coverage Report ---")
    coverage = analyze_quarterly_coverage(active_holdings_lf, temporal_bridge_lf)
    with pl.Config(tbl_rows=100):
        print(coverage)

    # 4. Extract Metadata (Lazy)
    bank_meta_lf, stock_meta_lf = extract_metadata(sec_root)

    # 5. Build Final GNN Tables
    edges_lf, banks_lf, stocks_lf = build_gnn_tables(
        active_holdings_lf, bank_meta_lf, stock_meta_lf
    )

    # 6. Build Stock-Stock Edges
    # IMPORTANT: Change 'Stock_symbol' to 'stock_id' here to match renamed edges_lf
    edges_ss_lf = build_stock_stock_edges(edges_lf, min_co_hold=2)

    print("\n--- Sinking GNN Tables to Parquet (Streaming Mode) ---")

    # Use sink_parquet for the largest files to protect your RAM
    try:
        # Stock-Stock is usually the one that crashes your PC
        edges_ss_lf.sink_parquet(output_path / "edges_stock_stock.parquet")

        # Bank-Stock is also quite large
        edges_lf.sink_parquet(output_path / "edges_bank_stock.parquet")

        # Nodes are usually small enough to collect
        banks_lf.collect().write_parquet(output_path / "nodes_bank.parquet")
        stocks_lf.collect().write_parquet(output_path / "nodes_stock.parquet")

    except pl.exceptions.ComputeError as e:
        print(f"Streaming failed, trying standard collect: {e}")
        # Fallback if your OS doesn't support streaming sinks for specific queries
        edges_ss_lf.collect().write_parquet(output_path / "edges_stock_stock.parquet")

    print("Done! Memory-safe processing complete.")
