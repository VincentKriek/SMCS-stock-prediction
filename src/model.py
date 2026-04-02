from lstm import LazyHeadlineVectorizer, LSTM_Encoder
from Mdgnn import MDGNN

from pathlib import Path
from typing import Optional

import polars as pl
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim



# Configuration
NEWS_N_ROWS = None

ROLLING_START_DATE = "2018-01-01"
ROLLING_END_DATE = "2023-02-28"
TRAIN_MONTHS = 6
VAL_MONTHS = 1
TEST_MONTHS = 6

HIDDEN_DIM = 128
GNN_LAYERS = 2
WINDOW_SIZE = 10
MAX_EPOCHS = 500
PATIENCE = 20
BATCH_SIZE = 8
DROPOUT = 0.1
MAX_SPLITS = None

TARGET_COL = "target_return"


NUMERIC_FEATURES = [
    "open",
    "high",
    # "low",
    # "close",
]

# Experiment setup
# - True / False  -> LSTM only
# - False / True  -> MDGNN only
# - True / True   -> LSTM + MDGNN
# - False / False -> invalid
USE_LSTM = True
USE_MDGNN = True


GRAPH_SPLIT_FILES = [
    "graphs_split_1.pt",
    "graphs_split_2.pt",
    "graphs_split_3.pt",
    "graphs_split_4.pt",
    "graphs_split_5.pt",
    "graphs_split_6.pt",
    "graphs_split_7.pt",
    "graphs_split_8.pt",
    "graphs_split_9.pt",
    "graphs_split_10.pt",
    "graphs_split_11.pt",
    "graphs_split_12.pt",
]



# Helper functions
def get_experiment_name(use_lstm: bool, use_mdgnn: bool) -> str:
    if use_lstm and use_mdgnn:
        return "lstm_mdgnn"
    if use_lstm and not use_mdgnn:
        return "lstm_only"
    if use_mdgnn and not use_lstm:
        return "mdgnn_only"
    raise ValueError("USE_LSTM and USE_MDGNN cannot both be False.")


def quarter_key_from_date(date_value) -> str:
    dt = pd.Timestamp(date_value)
    quarter = ((dt.month - 1) // 3) + 1
    return f"{dt.year}Q{quarter}"


def normalize_quarter_key(key) -> str:
    if isinstance(key, tuple) and len(key) == 2:
        return f"{int(key[0])}Q{int(key[1])}"
    return str(key).strip()


def _to_tensor(value, dtype=torch.float32) -> Optional[torch.Tensor]:
    if value is None:
        return None
    tensor = torch.as_tensor(value, dtype=dtype)
    return tensor


def _normalize_edge_value(edge_value):
    if edge_value is None:
        return None
    if not isinstance(edge_value, (list, tuple)) or len(edge_value) != 2:
        raise ValueError("Each edge entry must be [edge_index, edge_attr] or None.")

    edge_index = _to_tensor(edge_value[0], dtype=torch.long)
    edge_attr_raw = edge_value[1]
    edge_attr = None if edge_attr_raw is None else _to_tensor(edge_attr_raw, dtype=torch.float32)
    return (edge_index, edge_attr)


def normalize_snapshot_file_list(snapshot_files) -> list[str]:
    if snapshot_files is None:
        return []
    if isinstance(snapshot_files, (str, Path)):
        return [str(snapshot_files)]
    if isinstance(snapshot_files, (list, tuple)):
        files = [str(x) for x in snapshot_files if str(x).strip() != ""]
        return files
    raise ValueError("GRAPH_SPLIT_FILES must be a path string or a list of paths.")




def get_split_graph_files(split_idx: int, graph_split_files):
    file_list = normalize_snapshot_file_list(graph_split_files)
    if len(file_list) < 2:
        raise ValueError("At least two graph split files are required when USE_MDGNN=True.")

    train_pos = split_idx - 1
    eval_pos = split_idx
    if eval_pos >= len(file_list):
        raise IndexError(
            f"Split {split_idx} requires graph files at positions {train_pos} and {eval_pos}, "
            f"but only {len(file_list)} graph files were provided."
        )

    return file_list[train_pos], file_list[eval_pos]

def infer_graph_dims_from_snapshot_files(snapshot_files):
    raw_quarters = load_raw_snapshot_data(snapshot_files)
    if len(raw_quarters) == 0:
        raise ValueError("Graph snapshot files do not contain any quarters.")

    first_quarter = next(iter(raw_quarters.values()))
    stock_feat = _to_tensor(first_quarter.get("stock_feat"), dtype=torch.float32)
    bank_feat_raw = first_quarter.get("bank_feat")
    bank_feat = None if bank_feat_raw is None else _to_tensor(bank_feat_raw, dtype=torch.float32)
    industry_feat_raw = first_quarter.get("industry_feat")
    industry_feat = None if industry_feat_raw is None else _to_tensor(industry_feat_raw, dtype=torch.float32)

    edges = first_quarter.get("edges", {})
    edge_dims = {}
    for rel in ["SS", "SB", "BS", "SI", "IS", "II"]:
        value = edges.get(rel)
        if value is None:
            edge_dims[rel] = 0
        else:
            _, edge_attr = _normalize_edge_value(value)
            edge_dims[rel] = 0 if edge_attr is None else int(edge_attr.shape[-1])

    stock_in_dim = 0 if stock_feat is None else int(stock_feat.shape[-1])
    bank_in_dim = 0 if bank_feat is None else int(bank_feat.shape[-1])
    industry_in_dim = 1 if industry_feat is None else int(industry_feat.shape[-1])

    return stock_in_dim, bank_in_dim, industry_in_dim, edge_dims



# Graph snapshot loading
def load_single_snapshot_file(snapshot_file: str):
    path = Path(snapshot_file)
    if not path.exists():
        raise FileNotFoundError(f"Graph snapshot file not found: {snapshot_file}")

    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        loaded = torch.load(path, map_location="cpu")
    else:
        raise ValueError(f"Graph snapshot file must be .pt or .pth: {snapshot_file}")

    if isinstance(loaded, dict) and "cache" in loaded and isinstance(loaded["cache"], dict):
        loaded = loaded["cache"]

    if not isinstance(loaded, dict):
        raise ValueError(f"Graph snapshot file must contain a dict of quarter snapshots: {snapshot_file}")

    normalized = {}
    for quarter_id, snapshot in loaded.items():
        normalized[normalize_quarter_key(quarter_id)] = snapshot
    return normalized


def load_raw_snapshot_data(snapshot_files):
    file_list = normalize_snapshot_file_list(snapshot_files)
    if len(file_list) == 0:
        raise ValueError("GRAPH_SPLIT_FILES is empty.")

    merged = {}
    for snapshot_file in file_list:
        loaded = load_single_snapshot_file(snapshot_file)
        for quarter_id, snapshot in loaded.items():
            if quarter_id in merged:
                raise ValueError(
                    f"Duplicate quarter '{quarter_id}' found in multiple graph snapshot files."
                )
            merged[quarter_id] = snapshot
    return merged



class EmptyGraphFeatureCache:
    def __init__(self, hidden_dim: int = 128):
        self.hidden_dim = hidden_dim

    def lookup(self, date_value):
        return torch.zeros(self.hidden_dim, dtype=torch.float32)

    def lookup_window(self, date_value, window_size=10):
        return torch.zeros(window_size, self.hidden_dim, dtype=torch.float32)


class SnapshotGraphFeatureCache:
    def __init__(
        self,
        snapshot_files,
        mdgnn_model: MDGNN,
        device: torch.device,
        hidden_dim: int = 128,
    ):
        self.hidden_dim = hidden_dim
        self.device = device
        self.cache = {}
        self.available_quarters = []

        raw_quarters = load_raw_snapshot_data(snapshot_files)
        mdgnn_model = mdgnn_model.to(device)
        mdgnn_model.eval()

        with torch.no_grad():
            for quarter_id, snap in raw_quarters.items():
                stock_feat = _to_tensor(snap.get("stock_feat"), dtype=torch.float32)
                if stock_feat is None:
                    continue
                stock_feat = stock_feat.to(device)

                bank_feat_raw = snap.get("bank_feat")
                bank_feat = None if bank_feat_raw is None else _to_tensor(bank_feat_raw, dtype=torch.float32).to(device)

                industry_feat_raw = snap.get("industry_feat")
                industry_feat = None if industry_feat_raw is None else _to_tensor(industry_feat_raw, dtype=torch.float32).to(device)

                raw_edges = snap.get("edges", {})
                edges = {}
                for rel in ["SS", "SB", "BS", "SI", "IS", "II"]:
                    normalized_edge = _normalize_edge_value(raw_edges.get(rel))
                    if normalized_edge is None:
                        edges[rel] = None
                    else:
                        edge_index, edge_attr = normalized_edge
                        edges[rel] = (
                            edge_index.to(device),
                            None if edge_attr is None else edge_attr.to(device),
                        )

                stock_emb = mdgnn_model.encode_snapshot(
                    stock_feat=stock_feat,
                    bank_feat=bank_feat,
                    industry_feat=industry_feat,
                    edges=edges,
                )

                pooled = stock_emb.mean(dim=0).detach().cpu()
                self.cache[quarter_id] = pooled

        self.available_quarters = sorted(self.cache.keys())

    def lookup(self, date_value):
        q = quarter_key_from_date(date_value)
        if q not in self.cache:
            return torch.zeros(self.hidden_dim, dtype=torch.float32)
        return self.cache[q].clone().float()

    def lookup_window(self, date_value, window_size=10):
        target_date = pd.Timestamp(date_value).normalize()
        target_quarter = quarter_key_from_date(target_date)

        quarter_ends = []
        for quarter_id in self.available_quarters:
            year = int(quarter_id.split("Q")[0])
            quarter = int(quarter_id.split("Q")[1])
            end_month = quarter * 3
            quarter_end = pd.Timestamp(year=year, month=end_month, day=1) + pd.offsets.MonthEnd(0)
            if quarter_end <= target_date or quarter_id == target_quarter:
                quarter_ends.append((quarter_end, quarter_id))

        if len(quarter_ends) == 0:
            return torch.zeros(window_size, self.hidden_dim, dtype=torch.float32)

        quarter_ends = sorted(quarter_ends, key=lambda x: x[0])
        chosen_quarters = [q_id for _, q_id in quarter_ends[-window_size:]]
        seq = [self.cache[q_id].clone().float() for q_id in chosen_quarters]

        if len(seq) < window_size:
            pad_len = window_size - len(seq)
            pads = [torch.zeros(self.hidden_dim, dtype=torch.float32) for _ in range(pad_len)]
            seq = pads + seq

        return torch.stack(seq, dim=0)



# Dataset
class NewsGraphDataset(Dataset):
    def __init__(
        self,
        rows,
        stock2id: dict[str, int],
        graph_cache,
        embedding_col: str,
        feature_cols: list[str],
        target_col: str,
        max_headline_len: int,
        graph_hidden_dim: int = 128,
        window_size: int = 10,
        use_graph: bool = True,
    ):
        self.samples = []
        self.window_size = window_size
        self.graph_hidden_dim = graph_hidden_dim
        self.use_graph = use_graph
        self.feature_cols = feature_cols

        for row in rows:
            target = row.get(target_col)
            if target is None:
                continue

            stock_symbol = row.get("Stock_symbol")
            if stock_symbol not in stock2id:
                continue

            text_ids = row.get(embedding_col)
            if text_ids is None:
                text_ids = [0] * max_headline_len
            else:
                if len(text_ids) < max_headline_len:
                    text_ids = text_ids + [0] * (max_headline_len - len(text_ids))
                else:
                    text_ids = text_ids[:max_headline_len]

            numeric_feats = []
            for col in feature_cols:
                val = row.get(col)
                if val is None:
                    val = 0.0
                numeric_feats.append(float(val))

            if self.use_graph and graph_cache is not None:
                date_value = row.get("Date")
                graph_seq = graph_cache.lookup_window(date_value, window_size=window_size)
            else:
                graph_seq = torch.zeros(window_size, graph_hidden_dim, dtype=torch.float32)

            self.samples.append({
                "text_ids": torch.tensor(text_ids, dtype=torch.long),
                "numeric_feats": torch.tensor(numeric_feats, dtype=torch.float32),
                "target": torch.tensor(float(target), dtype=torch.float32),
                "stock_id": torch.tensor(stock2id[stock_symbol], dtype=torch.long),
                "graph_seq": graph_seq,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            s["text_ids"],
            s["numeric_feats"],
            s["target"],
            s["stock_id"],
            s["graph_seq"],
        )


# Models
class NumericFeatureProjector(nn.Module):
    def __init__(self, num_numeric_features: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.num_numeric_features = num_numeric_features
        if num_numeric_features > 0:
            self.proj = nn.Sequential(
                nn.Linear(num_numeric_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.proj = None

    def forward(self, numeric_feats: Optional[torch.Tensor], batch_size: int, device: torch.device):
        if self.proj is None:
            return torch.zeros(batch_size, 0, device=device)
        return self.proj(numeric_feats)


class LSTMOnlyModel(nn.Module):
    def __init__(
        self,
        lstm_encoder: LSTM_Encoder,
        num_numeric_features: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm_encoder = lstm_encoder
        self.numeric_proj = NumericFeatureProjector(num_numeric_features, hidden_dim, dropout)
        fusion_input_dim = hidden_dim + (hidden_dim if num_numeric_features > 0 else 0)

        self.head = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, text_ids, numeric_feats, stock_ids, graph_seq=None):
        text_repr = self.lstm_encoder(text_ids, stock_ids)
        parts = [text_repr]
        num_repr = self.numeric_proj(numeric_feats, text_ids.size(0), text_ids.device)
        if num_repr.shape[1] > 0:
            parts.append(num_repr)
        fused = torch.cat(parts, dim=1)
        return self.head(fused).squeeze(-1)


class MDGNNOnlyModel(nn.Module):
    def __init__(
        self,
        mdgnn_model: MDGNN,
        num_numeric_features: int,
        graph_hidden_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mdgnn_model = mdgnn_model
        self.numeric_proj = NumericFeatureProjector(num_numeric_features, hidden_dim, dropout)
        self.graph_proj = nn.Sequential(
            nn.Linear(graph_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        fusion_input_dim = hidden_dim + (hidden_dim if num_numeric_features > 0 else 0)

        self.head = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, text_ids=None, numeric_feats=None, stock_ids=None, graph_seq=None):
        _, graph_repr, _ = self.mdgnn_model.forward_from_sequence(graph_seq, return_attention=True)
        graph_repr = self.graph_proj(graph_repr)
        parts = [graph_repr]
        batch_size = graph_seq.size(0)
        device = graph_seq.device
        num_repr = self.numeric_proj(numeric_feats, batch_size, device)
        if num_repr.shape[1] > 0:
            parts.append(num_repr)
        fused = torch.cat(parts, dim=1)
        return self.head(fused).squeeze(-1)


class LSTM_MDGNN_Fusion(nn.Module):
    def __init__(
        self,
        lstm_encoder: LSTM_Encoder,
        mdgnn_model: MDGNN,
        num_numeric_features: int,
        graph_hidden_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm_encoder = lstm_encoder
        self.mdgnn_model = mdgnn_model
        self.numeric_proj = NumericFeatureProjector(num_numeric_features, hidden_dim, dropout)
        self.graph_proj = nn.Sequential(
            nn.Linear(graph_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        fusion_input_dim = hidden_dim + hidden_dim + (hidden_dim if num_numeric_features > 0 else 0)

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, text_ids, numeric_feats, stock_ids, graph_seq):
        text_repr = self.lstm_encoder(text_ids, stock_ids)
        _, graph_repr, _ = self.mdgnn_model.forward_from_sequence(graph_seq, return_attention=True)
        graph_repr = self.graph_proj(graph_repr)
        parts = [text_repr, graph_repr]
        num_repr = self.numeric_proj(numeric_feats, text_ids.size(0), text_ids.device)
        if num_repr.shape[1] > 0:
            parts.append(num_repr)
        fused = torch.cat(parts, dim=1)
        return self.fusion_head(fused).squeeze(-1)


def build_model(
    use_lstm: bool,
    use_mdgnn: bool,
    lstm_encoder: Optional[LSTM_Encoder],
    mdgnn_model: Optional[MDGNN],
    num_numeric_features: int,
    hidden_dim: int,
    dropout: float = 0.1,
):
    if use_lstm and use_mdgnn:
        return LSTM_MDGNN_Fusion(
            lstm_encoder=lstm_encoder,
            mdgnn_model=mdgnn_model,
            num_numeric_features=num_numeric_features,
            graph_hidden_dim=hidden_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    if use_lstm and not use_mdgnn:
        return LSTMOnlyModel(
            lstm_encoder=lstm_encoder,
            num_numeric_features=num_numeric_features,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    if use_mdgnn and not use_lstm:
        return MDGNNOnlyModel(
            mdgnn_model=mdgnn_model,
            num_numeric_features=num_numeric_features,
            graph_hidden_dim=hidden_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    raise ValueError("USE_LSTM and USE_MDGNN cannot both be False.")


# Data preparation
def add_next_day_return_target(lf: pl.LazyFrame) -> pl.LazyFrame:
    lf = lf.sort(["Stock_symbol", "Date"]).with_columns(
        (
            (pl.col("close").shift(-1).over("Stock_symbol") - pl.col("close"))
            / pl.col("close")
        ).alias("target_return")
    )
    lf = lf.filter(pl.col("target_return").is_not_null())
    return lf


def make_halfyear_rolling_splits(
    data: pl.LazyFrame,
    numeric_features: list[str],
    start_date: str = "2018-01-01",
    end_date: str = "2023-02-28",
    train_months: int = 6,
    val_months: int = 1,
    test_months: int = 6,
):
    schema_names = data.collect_schema().names()
    needed_cols = ["Date", "Stock_symbol", "embedded_headline", TARGET_COL] + numeric_features
    used_cols = [c for c in needed_cols if c in schema_names]

    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)

    df = (
        data
        .select(used_cols)
        .filter(
            (pl.col("Date") >= pl.lit(start_dt)) &
            (pl.col("Date") <= pl.lit(end_dt))
        )
        .sort("Date")
        .collect(engine="streaming")
        .to_pandas()
    )

    if len(df) == 0:
        return []

    df["Date"] = pd.to_datetime(df["Date"])

    splits = []
    anchor = start_dt
    while True:
        train_start = anchor
        train_end = train_start + pd.DateOffset(months=train_months) - pd.Timedelta(days=1)
        val_start = train_end + pd.Timedelta(days=1)
        val_end = val_start + pd.DateOffset(months=val_months) - pd.Timedelta(days=1)
        test_start = val_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)

        if val_start > end_dt or test_start > end_dt:
            break

        test_end = min(test_end, end_dt)

        train_df = df[(df["Date"] >= train_start) & (df["Date"] <= train_end)].copy()
        val_df = df[(df["Date"] >= val_start) & (df["Date"] <= val_end)].copy()
        test_df = df[(df["Date"] >= test_start) & (df["Date"] <= test_end)].copy()

        if len(train_df) > 0 and len(val_df) > 0 and len(test_df) > 0:
            splits.append({
                "train_rows": train_df.to_dict("records"),
                "val_rows": val_df.to_dict("records"),
                "test_rows": test_df.to_dict("records"),
            })

        anchor = anchor + pd.DateOffset(months=test_months)
        if anchor > end_dt:
            break

    return splits


def build_loaders_for_split(
    split_dict,
    stock2id,
    train_graph_cache,
    eval_graph_cache,
    batch_size,
    max_headline_len,
    feature_cols,
    target_col,
    window_size=10,
    use_graph=True,
):
    train_dataset = NewsGraphDataset(
        rows=split_dict["train_rows"],
        stock2id=stock2id,
        graph_cache=train_graph_cache,
        embedding_col="embedded_headline",
        feature_cols=feature_cols,
        target_col=target_col,
        max_headline_len=max_headline_len,
        graph_hidden_dim=HIDDEN_DIM,
        window_size=window_size,
        use_graph=use_graph,
    )

    val_dataset = NewsGraphDataset(
        rows=split_dict["val_rows"],
        stock2id=stock2id,
        graph_cache=eval_graph_cache,
        embedding_col="embedded_headline",
        feature_cols=feature_cols,
        target_col=target_col,
        max_headline_len=max_headline_len,
        graph_hidden_dim=HIDDEN_DIM,
        window_size=window_size,
        use_graph=use_graph,
    )

    test_dataset = NewsGraphDataset(
        rows=split_dict["test_rows"],
        stock2id=stock2id,
        graph_cache=eval_graph_cache,
        embedding_col="embedded_headline",
        feature_cols=feature_cols,
        target_col=target_col,
        max_headline_len=max_headline_len,
        graph_hidden_dim=HIDDEN_DIM,
        window_size=window_size,
        use_graph=use_graph,
    )

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )



# Main
if __name__ == "__main__":
    experiment_name = get_experiment_name(USE_LSTM, USE_MDGNN)
    print(f"Running experiment mode: {experiment_name}")
    print(f"Using numeric features: {NUMERIC_FEATURES}")
    print(f"Using ordered graph split files: {GRAPH_SPLIT_FILES}")

    l = LazyHeadlineVectorizer(
        "prepared_data_2018-01-01_2023-12-31.parquet",
        n_rows=NEWS_N_ROWS,
    )
    l.run()

    l.lf = add_next_day_return_target(l.lf)

    schema_names = l.lf.collect_schema().names()
    available_numeric_features = [c for c in NUMERIC_FEATURES if c in schema_names]
    missing_numeric_features = [c for c in NUMERIC_FEATURES if c not in schema_names]

    if missing_numeric_features:
        print(f"Warning: these numeric features were not found and will be ignored: {missing_numeric_features}")

    stocks = sorted(l.lf.select("Stock_symbol").unique().collect().to_series().to_list())
    stock2id = {symbol: idx for idx, symbol in enumerate(stocks)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mdgnn = None

    if USE_MDGNN:
        ordered_graph_files = normalize_snapshot_file_list(GRAPH_SPLIT_FILES)
        if len(ordered_graph_files) < 2:
            raise ValueError("At least two graph split files are required when USE_MDGNN=True.")

        stock_in_dim, bank_in_dim, industry_in_dim, edge_dims = infer_graph_dims_from_snapshot_files(ordered_graph_files[0])


        mdgnn = MDGNN(
            stock_in_dim=stock_in_dim,
            bank_in_dim=bank_in_dim,
            industry_in_dim=industry_in_dim,
            edge_dims=edge_dims,
            hidden_dim=HIDDEN_DIM,
            gnn_layers=GNN_LAYERS,
            num_heads=4,
            ff_dim=256,
            dropout=DROPOUT,
        ).to(device)

    rolling_splits = make_halfyear_rolling_splits(
        data=l.lf,
        numeric_features=available_numeric_features,
        start_date=ROLLING_START_DATE,
        end_date=ROLLING_END_DATE,
        train_months=TRAIN_MONTHS,
        val_months=VAL_MONTHS,
        test_months=TEST_MONTHS,
    )

    if len(rolling_splits) == 0:
        print("No valid rolling splits were created.")
        raise SystemExit

    if MAX_SPLITS is not None:
        rolling_splits = rolling_splits[:MAX_SPLITS]
        print(f"Iteration limit enabled: only running first {len(rolling_splits)} split(s).")

    if USE_MDGNN:
        max_supported_splits = len(normalize_snapshot_file_list(GRAPH_SPLIT_FILES)) - 1
        print(f"Graph-based iterations available from provided files: {max_supported_splits}")
        if len(rolling_splits) > max_supported_splits:
            print(
                f"Warning: {len(rolling_splits)} data splits were created, but only {max_supported_splits} graph-based "
                f"iterations are supported by the provided graph files. Extra data splits will be skipped."
            )

    for split_idx, split_info in enumerate(rolling_splits, start=1):
        print(f"Starting split {split_idx}...")

        train_graph_cache = EmptyGraphFeatureCache(hidden_dim=HIDDEN_DIM)
        eval_graph_cache = EmptyGraphFeatureCache(hidden_dim=HIDDEN_DIM)

        if USE_MDGNN:
            try:
                train_graph_file, eval_graph_file = get_split_graph_files(split_idx, GRAPH_SPLIT_FILES)
            except IndexError:
                print(f"Split {split_idx} does not have enough graph files available. Skipping.")
                continue

            print(f"Split {split_idx} graph files | train={train_graph_file} | val/test={eval_graph_file}")
            train_graph_cache = SnapshotGraphFeatureCache(
                snapshot_files=train_graph_file,
                mdgnn_model=mdgnn,
                device=device,
                hidden_dim=HIDDEN_DIM,
            )
            eval_graph_cache = SnapshotGraphFeatureCache(
                snapshot_files=eval_graph_file,
                mdgnn_model=mdgnn,
                device=device,
                hidden_dim=HIDDEN_DIM,
            )

        train_loader, val_loader, test_loader = build_loaders_for_split(
            split_dict=split_info,
            stock2id=stock2id,
            train_graph_cache=train_graph_cache,
            eval_graph_cache=eval_graph_cache,
            batch_size=BATCH_SIZE,
            max_headline_len=l.max_headline_len,
            feature_cols=available_numeric_features,
            target_col=TARGET_COL,
            window_size=WINDOW_SIZE,
            use_graph=USE_MDGNN,
        )

        if len(train_loader.dataset) == 0 or len(val_loader.dataset) == 0 or len(test_loader.dataset) == 0:
            print(f"Split {split_idx} is empty. Skipping.")
            continue

        lstm_encoder = None
        if USE_LSTM:
            lstm_encoder = LSTM_Encoder(
                vocab_size=len(l.word2id),
                embedding_dim=l.vector_size,
                hidden_dim=HIDDEN_DIM,
                embedding_matrix=l.embedding_matrix,
                num_stocks=len(stocks),
                stock_emb_dim=HIDDEN_DIM,
                layer_dim=1,
                device=device,
            ).to(device)

        current_mdgnn = mdgnn if USE_MDGNN else None

        model = build_model(
            use_lstm=USE_LSTM,
            use_mdgnn=USE_MDGNN,
            lstm_encoder=lstm_encoder,
            mdgnn_model=current_mdgnn,
            num_numeric_features=len(available_numeric_features),
            hidden_dim=HIDDEN_DIM,
            dropout=DROPOUT,
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

        best_val_loss = float("inf")
        counter = 0
        save_path = f"best_{experiment_name}_split_{split_idx}.pt"

        for epoch in range(MAX_EPOCHS):
            model.train()
            train_loss = 0.0

            for X_text_batch, X_num_batch, Y_batch, stock_batch, graph_seq_batch in train_loader:
                X_text_batch = X_text_batch.to(device)
                X_num_batch = X_num_batch.to(device)
                Y_batch = Y_batch.to(device)
                stock_batch = stock_batch.to(device)
                graph_seq_batch = graph_seq_batch.to(device)

                optimizer.zero_grad()
                preds = model(
                    text_ids=X_text_batch,
                    numeric_feats=X_num_batch,
                    stock_ids=stock_batch,
                    graph_seq=graph_seq_batch,
                )

                loss = criterion(preds, Y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_text_val, X_num_val, Y_val, stock_val, graph_seq_val in val_loader:
                    X_text_val = X_text_val.to(device)
                    X_num_val = X_num_val.to(device)
                    Y_val = Y_val.to(device)
                    stock_val = stock_val.to(device)
                    graph_seq_val = graph_seq_val.to(device)

                    preds = model(
                        text_ids=X_text_val,
                        numeric_feats=X_num_val,
                        stock_ids=stock_val,
                        graph_seq=graph_seq_val,
                    )
                    val_loss += criterion(preds, Y_val).item()

            avg_train_loss = train_loss / max(len(train_loader), 1)
            avg_val_loss = val_loss / max(len(val_loader), 1)
            print(f"{experiment_name} | Split {split_idx} Epoch {epoch + 1} | Train={avg_train_loss:.6f} | Val={avg_val_loss:.6f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
                torch.save(model.state_dict(), save_path)
            else:
                counter += 1
                if counter >= PATIENCE:
                    break

        model.load_state_dict(torch.load(save_path, map_location=device))
        model.eval()

        test_loss = 0.0
        all_preds = []
        with torch.no_grad():
            for X_text_test, X_num_test, Y_test, stock_test, graph_seq_test in test_loader:
                X_text_test = X_text_test.to(device)
                X_num_test = X_num_test.to(device)
                Y_test = Y_test.to(device)
                stock_test = stock_test.to(device)
                graph_seq_test = graph_seq_test.to(device)

                preds = model(
                    text_ids=X_text_test,
                    numeric_feats=X_num_test,
                    stock_ids=stock_test,
                    graph_seq=graph_seq_test,
                )

                test_loss += criterion(preds, Y_test).item()
                all_preds.extend(preds.detach().cpu().tolist())

        avg_test_loss = test_loss / max(len(test_loader), 1)
        print(f"{experiment_name} | Split {split_idx} | Test={avg_test_loss:.6f}")
        print(f"{experiment_name} | Split {split_idx} | Predictions={all_preds}")



