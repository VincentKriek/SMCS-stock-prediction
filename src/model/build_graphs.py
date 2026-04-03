# TODO: ensure the stock nodes can be mapped to the stocks in the feature data

import polars as pl
import pandas as pd
import numpy as np
import torch
import os

from model import add_next_day_return_target

NEWS_N_ROWS = None
ROLLING_START_DATE = pd.Timestamp("2018-01-01")
ROLLING_END_DATE = pd.Timestamp("2023-12-31")
SPLIT_MONTHS = 6


def build_split_graphs(
    split_idx: int,
    split_info: dict,
    nodes_bank_path: str,
    nodes_stock_path: str,
    edges_bank_stock_path: str,
    edges_stock_stock_path: str,
    save_dir: str,
):
    print(f"\n--- Building Graphs for Split {split_idx} ---")

    # 1. Determine the start and end date for this split
    lf = split_info["lf"]
    if lf is None:
        print(f"Split {split_idx} is empty.")
        return

    trading_dates_df = lf.select("Date").unique().sort("Date").collect()

    trading_dates = trading_dates_df["Date"].to_list()
    if not trading_dates:
        return

    min_date, max_date = trading_dates[0], trading_dates[-1]

    print(
        f"Split {split_idx} range: {min_date.date()} to {max_date.date()}, {len(trading_dates)} trading days."
    )

    # 2. Determine needed quarters
    needed_quarters = sorted(
        {(ts.year, ((ts.month - 1) // 3 + 1)) for ts in trading_dates}
    )

    quarter_expr = None
    for y, q in needed_quarters:
        cond = (pl.col("year") == y) & (pl.col("quarter") == q)
        quarter_expr = cond if quarter_expr is None else (quarter_expr | cond)

    if quarter_expr is None:
        print("No quarters to load.")
        return

    # 3. Read only required node columns
    stock_cols = [
        "year",
        "quarter",
        "stock_id",
        "num_holders",
        "total_institutional_value",
        "total_institutional_shares",
        "num_quarters_held",
    ]
    bank_cols = [
        "year",
        "quarter",
        "bank_id",
        "num_stocks_held",
        "total_aum_value",
        "avg_position_size",
        "num_quarters_active",
    ]

    print("Loading stock node data...")
    nodes_stock_lf = (
        pl.scan_parquet(nodes_stock_path).select(stock_cols).filter(quarter_expr)
    )

    print("Loading bank node data...")
    nodes_bank_lf = (
        pl.scan_parquet(nodes_bank_path).select(bank_cols).filter(quarter_expr)
    )

    if (
        nodes_stock_lf.select(pl.len()).collect().item() == 0
        or nodes_bank_lf.select(pl.len()).collect().item() == 0
    ):
        print("Empty node data for this split.")
        return

    # 4. Read only required edge columns
    edge_bs_cols = [
        "year",
        "quarter",
        "bank_id",
        "stock_id",
        "total_value",
        "total_shares",
        "voting_sole",
        "voting_shared",
        "voting_none",
    ]
    edge_ss_cols = [
        "year",
        "quarter",
        "stock_id_1",
        "stock_id_2",
        "co_holder_count",
    ]

    print("Loading bank-stock edge data...")
    edges_bank_stock_lf = (
        pl.scan_parquet(edges_bank_stock_path).select(edge_bs_cols).filter(quarter_expr)
    )

    print("Loading stock-stock edge data...")
    edges_stock_stock_lf = (
        pl.scan_parquet(edges_stock_stock_path)
        .select(edge_ss_cols)
        .filter(quarter_expr)
    )

    # 5. Build quarter snapshots
    print("Building quarter snapshots...")
    quarter_snapshots = build_quarter_snapshots(
        nodes_bank_lf=nodes_bank_lf,
        nodes_stock_lf=nodes_stock_lf,
        edges_bank_stock_lf=edges_bank_stock_lf,
        edges_stock_stock_lf=edges_stock_stock_lf,
    )

    # 6. Save directly to .pt file without encoding via MDGNN
    save_path = os.path.join(save_dir, f"graphs_split_{split_idx}.pt")
    os.makedirs(save_dir, exist_ok=True)

    print(f"Saving {len(quarter_snapshots)} quarterly snapshots to {save_path} ...")
    torch.save(quarter_snapshots, save_path)
    print("Saved successfully.")


def make_splits(
    data: pl.LazyFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    split_months: int,
):
    splits = []
    split_start = start_date

    print("Generating lazy split plans")
    while split_start <= end_date:
        split_end = split_start + pd.DateOffset(months=split_months)

        split_plan = data.filter(
            (pl.col("Date") >= split_start) & (pl.col("Date") < split_end)
        )

        splits.append({"lf": split_plan, "date_range": (split_start, split_end)})

        split_start = split_end

    return splits


def build_quarter_snapshots(
    nodes_bank_lf: pl.LazyFrame,
    nodes_stock_lf: pl.LazyFrame,
    edges_bank_stock_lf: pl.LazyFrame,
    edges_stock_stock_lf: pl.LazyFrame,
):
    snapshots = {}

    quarter_pairs = (
        nodes_stock_lf.select(["year", "quarter"])
        .unique()
        .sort(["year", "quarter"])
        .collect()
    )

    for row in quarter_pairs.iter_rows(named=True):
        y, q = row["year"], row["quarter"]
        q_label = f"{y}Q{q}"

        stock_nodes = (
            nodes_stock_lf.filter((pl.col("year") == y) & (pl.col("quarter") == q))
            .sort("stock_id")
            .with_row_index("s_idx")
        )
        bank_nodes = (
            nodes_bank_lf.filter((pl.col("year") == y) & (pl.col("quarter") == q))
            .sort("bank_id")
            .with_row_index("b_idx")
        )

        bank_stock_edges = (
            edges_bank_stock_lf.filter((pl.col("year") == y) & (pl.col("quarter") == q))
            .join(bank_nodes.select(["bank_id", "b_idx"]), on="bank_id")
            .join(stock_nodes.select(["stock_id", "s_idx"]), on="stock_id")
        )

        stock_stock_edges = (
            edges_stock_stock_lf.filter(
                (pl.col("year") == y) & (pl.col("quarter") == q)
            )
            .join(
                stock_nodes.select(
                    [
                        pl.col("stock_id").alias("stock_id_1"),
                        pl.col("s_idx").alias("s_idx_1"),
                    ]
                ),
                on="stock_id_1",
            )
            .join(
                stock_nodes.select(
                    [
                        pl.col("stock_id").alias("stock_id_2"),
                        pl.col("s_idx").alias("s_idx_2"),
                    ]
                ),
                on="stock_id_2",
            )
        )

        print(f"Streaming data for {q_label}...")
        s_df, b_df, bs_df, ss_df = pl.collect_all(
            [stock_nodes, bank_nodes, bank_stock_edges, stock_stock_edges],
            engine="streaming",
        )

        # 5. Conversion to Tensors
        stock_feat_cols = [
            "num_holders",
            "total_institutional_value",
            "total_institutional_shares",
            "num_quarters_held",
        ]
        bank_feat_cols = [
            "num_stocks_held",
            "total_aum_value",
            "avg_position_size",
            "num_quarters_active",
        ]
        bs_feat_cols = [
            "total_value",
            "total_shares",
            "voting_sole",
            "voting_shared",
            "voting_none",
        ]
        ss_feat_cols = ["co_holder_count"]

        edge_index_stock_bank = torch.tensor(
            np.vstack([bs_df["b_idx"].to_numpy(), bs_df["s_idx"].to_numpy()]),
            dtype=torch.long,
        )
        edge_attr_stock_bank = torch.tensor(
            bs_df.select(bs_feat_cols).fill_null(0.0).to_numpy(), dtype=torch.float32
        )

        edge_index_stock_stock = torch.tensor(
            np.vstack([ss_df["s_idx_1"].to_numpy(), ss_df["s_idx_2"].to_numpy()]),
            dtype=torch.long,
        )

        edge_attr_stock_stock = torch.tensor(
            ss_df.select(ss_feat_cols).fill_null(0.0).to_numpy(),
            dtype=torch.float32,
        )

        # 6. Store Snapshot
        snapshots[q_label] = {
            "stock_ids": s_df["stock_id"].to_list(),
            "bank_ids": b_df["bank_id"].to_list(),
            "stock_feat": torch.tensor(
                s_df.select(stock_feat_cols).fill_null(0.0).to_numpy(),
                dtype=torch.float32,
            ),
            "bank_feat": torch.tensor(
                b_df.select(bank_feat_cols).fill_null(0.0).to_numpy(),
                dtype=torch.float32,
            ),
            "edges": {
                "SB": (edge_index_stock_bank, edge_attr_stock_bank),
                "BS": (
                    torch.stack([edge_index_stock_bank[1], edge_index_stock_bank[0]]),
                    edge_attr_stock_bank.clone(),
                ),
                "SS": (edge_index_stock_stock, edge_attr_stock_stock),
            },
        }

    return snapshots


def main():
    data_path = "data/pre-processor/prepared_data_2018-01-01_2023-12-31.parquet"

    print("Scanning data for trading dates and target alignment...")
    lf = (
        pl.scan_parquet(data_path)
        .select(["Date", "Stock_symbol", "close"])
        .pipe(add_next_day_return_target)
    )

    print("Generating rolling splits based on data...")
    rolling_splits = make_splits(
        data=lf,
        start_date=ROLLING_START_DATE,
        end_date=ROLLING_END_DATE,
        split_months=SPLIT_MONTHS,
    )

    if len(rolling_splits) == 0:
        print("No valid rolling splits were created.")
        return

    save_dir = "data/model/graphs"
    for split_idx, split_info in enumerate(rolling_splits, start=1):
        build_split_graphs(
            split_idx=split_idx,
            split_info=split_info,
            nodes_bank_path="data/graphs/nodes_bank.parquet",
            nodes_stock_path="data/graphs/nodes_stock.parquet",
            edges_bank_stock_path="data/graphs/edges_bank_stock.parquet",
            edges_stock_stock_path="data/graphs/edges_stock_stock.parquet",
            save_dir=save_dir,
        )


if __name__ == "__main__":
    main()
