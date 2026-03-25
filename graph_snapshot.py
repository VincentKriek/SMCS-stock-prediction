import pandas as pd
import torch


def quarter_key(year: int, quarter: int) -> str:
    return f"{int(year)}Q{int(quarter)}"


def get_quarter_from_date(dt) -> str:
    month = dt.month
    quarter = (month - 1) // 3 + 1
    return f"{dt.year}Q{quarter}"


# Build quarter-level graph snapshots
def build_quarter_snapshots(nodes_bank_df, nodes_stock_df, edges_df):
    snapshots = {}

    quarter_pairs = sorted(set(zip(nodes_stock_df["year"], nodes_stock_df["quarter"])))

    for year, quarter in quarter_pairs:
        bank_df_q = nodes_bank_df[
            (nodes_bank_df["year"] == year) & (nodes_bank_df["quarter"] == quarter)
        ].copy()

        stock_df_q = nodes_stock_df[
            (nodes_stock_df["year"] == year) & (nodes_stock_df["quarter"] == quarter)
        ].copy()

        edge_df_q = edges_df[
            (edges_df["year"] == year) & (edges_df["quarter"] == quarter)
        ].copy()

        bank_ids = sorted(bank_df_q["bank_id"].astype(str).unique())
        stock_ids = sorted(stock_df_q["stock_id"].astype(str).unique())

        bank2idx = {x: i for i, x in enumerate(bank_ids)}
        stock2idx = {x: i for i, x in enumerate(stock_ids)}

        edge_df_q = edge_df_q[
            edge_df_q["bank_id"].astype(str).isin(bank2idx) &
            edge_df_q["stock_id"].astype(str).isin(stock2idx)
        ].copy()

        bank_feat_cols = [
            "num_stocks_held",
            "total_aum_value",
            "avg_position_size",
            "num_quarters_active"
        ]
        stock_feat_cols = [
            "num_holders",
            "total_institutional_value",
            "total_institutional_shares",
            "num_quarters_held"
        ]
        edge_feat_cols = [
            "total_value",
            "total_shares",
            "voting_sole",
            "voting_shared",
            "voting_none"
        ]

        bank_feat = torch.tensor(
            bank_df_q[bank_feat_cols].fillna(0.0).to_numpy(),
            dtype=torch.float32
        )
        stock_feat = torch.tensor(
            stock_df_q[stock_feat_cols].fillna(0.0).to_numpy(),
            dtype=torch.float32
        )

        # SB: bank -> stock
        edge_index_sb = torch.tensor([
            edge_df_q["bank_id"].astype(str).map(bank2idx).to_numpy(),
            edge_df_q["stock_id"].astype(str).map(stock2idx).to_numpy()
        ], dtype=torch.long)

        edge_attr_sb = torch.tensor(
            edge_df_q[edge_feat_cols].fillna(0.0).to_numpy(),
            dtype=torch.float32
        )

        # BS: stock -> bank (reverse edge)
        edge_index_bs = torch.stack([edge_index_sb[1], edge_index_sb[0]], dim=0)
        edge_attr_bs = edge_attr_sb.clone()

        snapshots[quarter_key(year, quarter)] = {
            "bank_ids": bank_ids,
            "stock_ids": stock_ids,
            "stock2idx": stock2idx,
            "bank2idx": bank2idx,
            "stock_feat": stock_feat,
            "bank_feat": bank_feat,
            "industry_feat": None,
            "edges": {
                "SS": None,
                "SB": (edge_index_sb, edge_attr_sb),
                "BS": (edge_index_bs, edge_attr_bs),
                "SI": None,
                "IS": None,
                "II": None,
            }
        }

    return snapshots


# Map each trading day to the corresponding quarter snapshot
def expand_quarter_snapshots_to_daily(trading_dates, quarter_snapshots):
    daily_snapshots = {}

    for dt in trading_dates:
        q = get_quarter_from_date(pd.Timestamp(dt))
        if q in quarter_snapshots:
            daily_snapshots[pd.Timestamp(dt)] = quarter_snapshots[q]

    return daily_snapshots