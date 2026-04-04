from dotenv import load_dotenv

load_dotenv()

from lstm import LazyHeadlineVectorizer, LSTM_Encoder  # noqa: E402
from Mdgnn import MDGNN  # noqa: E402

from pathlib import Path  # noqa: E402
from typing import Optional  # noqa: E402

import polars as pl  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
import torch.optim as optim  # noqa: E402
import re  # noqa: E402
import gc  # noqa: E402
from tqdm import tqdm  # noqa: E402


# Configuration
NEWS_N_ROWS = None

ROLLING_START_DATE = "2018-01-01"
ROLLING_END_DATE = "2023-12-31"
TRAIN_MONTHS = 6
VAL_MONTHS = 1
TEST_MONTHS = 5

HIDDEN_DIM = 128
GNN_LAYERS = 2
WINDOW_SIZE = 10
MAX_EPOCHS = 500
PATIENCE = 20
BATCH_SIZE = 8
DROPOUT = 0.1
MAX_SPLITS = None

TARGET_COL = "target_return"

# all columns:
# ['Date', 'open', 'high', 'low', 'close', 'adj close', 'volume', 'Stock_symbol', 'row_index', 'Article_title', 'summary', 'Sentiment_llm_mean_filled', 'Sentiment_llm_median_filled', 'Sentiment_llm_mode_filled', 'tokenized_headline', 'headline_len', 'embedded_headline', 'target_return']
NUMERIC_FEATURES = ["open", "high", "low", "close", "adj close", "volume"]

# Experiment setup
# - True / False  -> LSTM only
# - False / True  -> MDGNN only
# - True / True   -> LSTM + MDGNN
# - False / False -> invalid
USE_LSTM = True
USE_MDGNN = True
LLM_SENTIMENT_MODE = "mean"  # "mean", "median" or "mode". Use None to exclude column

GRAPH_SPLIT_BASE_PATH = "data/model/graphs"
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

CUSIP_MAPPING_FILE = "CUSIP.csv"


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
    edge_attr = (
        None
        if edge_attr_raw is None
        else _to_tensor(edge_attr_raw, dtype=torch.float32)
    )
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
        raise ValueError(
            "At least two graph split files are required when USE_MDGNN=True."
        )

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
    bank_feat = (
        None
        if bank_feat_raw is None
        else _to_tensor(bank_feat_raw, dtype=torch.float32)
    )
    industry_feat_raw = first_quarter.get("industry_feat")
    industry_feat = (
        None
        if industry_feat_raw is None
        else _to_tensor(industry_feat_raw, dtype=torch.float32)
    )

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


def _normalize_stock_key(value) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


def _candidate_stock_keys(stock_symbol) -> list[str]:
    base = _normalize_stock_key(stock_symbol)
    if base == "":
        return []

    candidates = [base]

    if "." in base:
        candidates.append(base.replace(".", ""))
        candidates.append(base.replace(".", "-"))

    if "-" in base:
        candidates.append(base.replace("-", ""))
        candidates.append(base.replace("-", "."))

    return list(dict.fromkeys([x for x in candidates if x != ""]))


def load_cusip_symbol_mapping(mapping_file: str) -> dict[str, str]:
    path = Path(mapping_file)
    if not path.exists():
        print(f"Warning: CUSIP mapping file not found: {mapping_file}")
        return {}

    df = pd.read_csv(path, dtype=str)
    required_cols = {"cusip", "symbol"}
    missing_cols = required_cols - set(df.columns.str.lower())
    lower_to_original = {c.lower(): c for c in df.columns}

    if missing_cols:
        raise ValueError(
            f"CUSIP mapping file must contain columns {sorted(required_cols)}, "
            f"but found {list(df.columns)}"
        )

    cusip_col = lower_to_original["cusip"]
    symbol_col = lower_to_original["symbol"]

    mapping = {}
    for _, row in df[[cusip_col, symbol_col]].dropna().iterrows():
        cusip_raw = str(row[cusip_col]).strip().upper()
        symbol_raw = _normalize_stock_key(row[symbol_col])
        if cusip_raw == "" or symbol_raw == "":
            continue

        mapping[cusip_raw] = symbol_raw
        mapping[cusip_raw.lstrip("0")] = symbol_raw

    return mapping


def _map_graph_id_to_stock_symbol(graph_id, graph_cusip=None) -> str:
    for raw in [graph_cusip, graph_id]:
        if raw is None:
            continue

        raw_key = str(raw).strip().upper()
        if raw_key == "":
            continue

        if raw_key in GRAPH_ID_TO_SYMBOL:
            return _normalize_stock_key(GRAPH_ID_TO_SYMBOL[raw_key])

        stripped = raw_key.lstrip("0")
        if stripped in GRAPH_ID_TO_SYMBOL:
            return _normalize_stock_key(GRAPH_ID_TO_SYMBOL[stripped])

        if stripped != "" and re.fullmatch(r"[A-Z.\-]+", stripped):
            return stripped

        m = re.search(r"([A-Z.\-]+)$", stripped)
        if m:
            return m.group(1)

    return ""


GRAPH_ID_TO_SYMBOL = load_cusip_symbol_mapping(CUSIP_MAPPING_FILE)


# Graph snapshot loading
def load_single_snapshot_file(snapshot_file: str):
    path = Path(GRAPH_SPLIT_BASE_PATH).joinpath(snapshot_file)
    if not path.exists():
        raise FileNotFoundError(f"Graph snapshot file not found: {snapshot_file}")

    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        loaded = torch.load(path, map_location="cpu")
    else:
        raise ValueError(f"Graph snapshot file must be .pt or .pth: {snapshot_file}")

    if (
        isinstance(loaded, dict)
        and "cache" in loaded
        and isinstance(loaded["cache"], dict)
    ):
        loaded = loaded["cache"]

    if not isinstance(loaded, dict):
        raise ValueError(
            f"Graph snapshot file must contain a dict of quarter snapshots: {snapshot_file}"
        )

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

    def lookup(self, date_value, stock_symbol=None):
        return torch.zeros(self.hidden_dim, dtype=torch.float32)

    def lookup_window(self, date_value, stock_symbol=None, window_size=10):
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
        self.quarter_metadata = []

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
                bank_feat = (
                    None
                    if bank_feat_raw is None
                    else _to_tensor(bank_feat_raw, dtype=torch.float32).to(device)
                )

                industry_feat_raw = snap.get("industry_feat")
                industry_feat = (
                    None
                    if industry_feat_raw is None
                    else _to_tensor(industry_feat_raw, dtype=torch.float32).to(device)
                )

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

                stock_emb = (
                    mdgnn_model.encode_snapshot(
                        stock_feat=stock_feat,
                        bank_feat=bank_feat,
                        industry_feat=industry_feat,
                        edges=edges,
                    )
                    .detach()
                    .cpu()
                )

                quarter_cache = {}

                graph_stock_ids = snap.get("stock_ids", [])
                graph_stock_cusips = snap.get("stock_cusips", [])

                for idx in range(len(graph_stock_ids)):
                    emb = stock_emb[idx].float()

                    graph_id = (
                        graph_stock_ids[idx] if idx < len(graph_stock_ids) else None
                    )
                    graph_cusip = (
                        graph_stock_cusips[idx]
                        if idx < len(graph_stock_cusips)
                        else None
                    )

                    mapped_symbol = _map_graph_id_to_stock_symbol(graph_id, graph_cusip)
                    if mapped_symbol != "":
                        quarter_cache[mapped_symbol] = emb

                self.available_quarters = sorted(self.cache.keys())

        for q_id in self.available_quarters:
            year = int(q_id.split("Q")[0])
            q_num = int(q_id.split("Q")[1])
            q_end = pd.Timestamp(
                year=year, month=q_num * 3, day=1
            ) + pd.offsets.MonthEnd(0)

            self.quarter_metadata.append((q_end, q_id))

    def lookup(self, date_value, stock_symbol):
        q = quarter_key_from_date(date_value)
        if q not in self.cache:
            return torch.zeros(self.hidden_dim, dtype=torch.float32)

        quarter_cache = self.cache[q]
        for key in _candidate_stock_keys(stock_symbol):
            if key in quarter_cache:
                return quarter_cache[key].clone().float()

        return torch.zeros(self.hidden_dim, dtype=torch.float32)

    def lookup_window(self, date_value, stock_symbol, window_size=10):
        target_date = pd.Timestamp(date_value).normalize()
        target_quarter = quarter_key_from_date(target_date)

        quarter_ends = [
            (q_end, q_id)
            for q_end, q_id in self.quarter_metadata
            if q_end <= target_date or q_id == target_quarter
        ]

        if not quarter_ends:
            return torch.zeros(window_size, self.hidden_dim, dtype=torch.float32)

        quarter_ends.sort(key=lambda x: x[0])
        chosen_quarters = [q_id for _, q_id in quarter_ends[-window_size:]]
        candidates = _candidate_stock_keys(stock_symbol)

        seq = []
        for q_id in chosen_quarters:
            quarter_cache = self.cache.get(q_id, {})
            found = None
            for key in candidates:
                if key in quarter_cache:
                    found = quarter_cache[key]
                    break

            if found is None:
                seq.append(torch.zeros(self.hidden_dim, dtype=torch.float32))
            else:
                seq.append(found.clone().float())

        if len(seq) < window_size:
            pad_len = window_size - len(seq)
            pads = [torch.zeros(self.hidden_dim, dtype=torch.float32)] * pad_len
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
        self.valid_rows = []
        for row in rows:
            if row.get(target_col) is not None and row.get("Stock_symbol") in stock2id:
                self.valid_rows.append(row)

        self.stock2id = stock2id
        self.graph_cache = graph_cache
        self.embedding_col = embedding_col
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.max_headline_len = max_headline_len
        self.graph_hidden_dim = graph_hidden_dim
        self.window_size = window_size
        self.use_graph = use_graph

    def __len__(self):
        return len(self.valid_rows)

    def __getitem__(self, idx):
        row = self.valid_rows[idx]
        stock_symbol = row.get("Stock_symbol")

        raw_text_ids = row.get(self.embedding_col)

        if raw_text_ids is None:
            text_ids = [0] * self.max_headline_len
        else:
            text_ids = list(raw_text_ids)[: self.max_headline_len]
            text_ids += [0] * (self.max_headline_len - len(text_ids))

        numeric_feats = [
            (
                v if (v := float(row.get(col, 0.0) or 0.0)) == v else 0.0
            )  # replace NaN with 0.0
            for col in self.feature_cols
        ]

        if self.use_graph and self.graph_cache is not None:
            graph_seq = self.graph_cache.lookup_window(
                row.get("Date"),
                stock_symbol,
                window_size=self.window_size,
            )
        else:
            graph_seq = torch.zeros(
                self.window_size, self.graph_hidden_dim, dtype=torch.float32
            )

        return (
            torch.tensor(text_ids, dtype=torch.long),
            torch.tensor(numeric_feats, dtype=torch.float32),
            torch.tensor(float(row[self.target_col]), dtype=torch.float32),
            torch.tensor(self.stock2id[stock_symbol], dtype=torch.long),
            graph_seq,
        )


# Models
class NumericFeatureProjector(nn.Module):
    def __init__(
        self, num_numeric_features: int, hidden_dim: int = 128, dropout: float = 0.1
    ):
        super().__init__()
        self.num_numeric_features = num_numeric_features
        if num_numeric_features > 0:
            self.proj = nn.Sequential(
                nn.BatchNorm1d(num_numeric_features),
                nn.Linear(num_numeric_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.proj = None

    def forward(
        self,
        numeric_feats: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ):
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
        self.numeric_proj = NumericFeatureProjector(
            num_numeric_features, hidden_dim, dropout
        )
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
        self.numeric_proj = NumericFeatureProjector(
            num_numeric_features, hidden_dim, dropout
        )
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

    def forward(
        self, text_ids=None, numeric_feats=None, stock_ids=None, graph_seq=None
    ):
        _, graph_repr, _ = self.mdgnn_model.forward(graph_seq, return_attention=True)
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
        self.numeric_proj = NumericFeatureProjector(
            num_numeric_features, hidden_dim, dropout
        )
        self.graph_proj = nn.Sequential(
            nn.Linear(graph_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        fusion_input_dim = (
            hidden_dim + hidden_dim + (hidden_dim if num_numeric_features > 0 else 0)
        )

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, text_ids, numeric_feats, stock_ids, graph_seq):
        text_repr = self.lstm_encoder(text_ids, stock_ids)
        _, graph_repr, _ = self.mdgnn_model.forward(graph_seq, return_attention=True)
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
    lf = lf.filter(
        pl.col("target_return").is_not_null() & pl.col("target_return").is_finite()
    )
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
    needed_cols = [
        "Date",
        "Stock_symbol",
        "embedded_headline",
        TARGET_COL,
    ] + numeric_features
    used_cols = [c for c in needed_cols if c in schema_names]

    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)

    df = (
        data.select(used_cols)
        .filter(
            (pl.col("Date") >= pl.lit(start_dt)) & (pl.col("Date") <= pl.lit(end_dt))
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
        train_end = (
            train_start + pd.DateOffset(months=train_months) - pd.Timedelta(days=1)
        )
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
            splits.append(
                {
                    "train_rows": train_df.to_dict("records"),
                    "val_rows": val_df.to_dict("records"),
                    "test_rows": test_df.to_dict("records"),
                }
            )

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
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )


# Main
if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)  # for debug

    experiment_name = get_experiment_name(USE_LSTM, USE_MDGNN)
    print(f"Running experiment mode: {experiment_name}")
    print(f"Using numeric features: {NUMERIC_FEATURES}")
    print(f"Using LLM sentiment aggregation mode: {LLM_SENTIMENT_MODE}")
    print(f"Using ordered graph split files: {GRAPH_SPLIT_FILES}")
    print(f"Loaded CUSIP-symbol mappings: {len(GRAPH_ID_TO_SYMBOL)}")

    # path = "../../data/pre-processor/prepared_data_2018-01-01_2023-12-31.parquet"
    path = "data/pre-processor/prepared_data_2018-01-01_2023-12-31.parquet"
    prev_emb_file = Path(f"data/pre-processor/lf_tokenized_{NEWS_N_ROWS}_rows.parquet")
    print(f"Tokenized lf path exists?: {prev_emb_file.exists()}")

    prev_emb_file = prev_emb_file if prev_emb_file.exists() else None

    lhv = LazyHeadlineVectorizer(path, n_rows=NEWS_N_ROWS, prev_emb_file=prev_emb_file)
    lhv.run()

    lhv.lf = add_next_day_return_target(lhv.lf)

    # ['Date', 'open', 'high', 'low', 'close', 'adj close', 'volume', 'Stock_symbol', 'row_index', 'Article_title', 'summary', 'Sentiment_llm_mean_filled', 'Sentiment_llm_median_filled', 'Sentiment_llm_mode_filled', 'tokenized_headline', 'headline_len', 'embedded_headline', 'target_return']
    schema_names = lhv.lf.collect_schema().names()

    numeric_cols = NUMERIC_FEATURES
    match LLM_SENTIMENT_MODE:
        case "mean":
            numeric_cols.append("Sentiment_llm_mean_filled")
        case "median":
            numeric_cols.append("Sentiment_llm_median_filled")
        case "mode":
            numeric_cols.append("Sentiment_llm_mode_filled")

    available_numeric_features = [c for c in NUMERIC_FEATURES if c in schema_names]
    missing_numeric_features = [c for c in NUMERIC_FEATURES if c not in schema_names]

    if missing_numeric_features:
        print(
            f"Warning: these numeric features were not found and will be ignored: {missing_numeric_features}"
        )

    stocks = sorted(
        lhv.lf.select("Stock_symbol").unique().collect().to_series().to_list()
    )
    stock2id = {symbol: idx for idx, symbol in enumerate(stocks)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mdgnn = None

    if USE_MDGNN:
        ordered_graph_files = normalize_snapshot_file_list(GRAPH_SPLIT_FILES)
        if len(ordered_graph_files) < 2:
            raise ValueError(
                "At least two graph split files are required when USE_MDGNN=True."
            )

        stock_in_dim, bank_in_dim, industry_in_dim, edge_dims = (
            infer_graph_dims_from_snapshot_files(ordered_graph_files[0])
        )

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
        data=lhv.lf,
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
        print(
            f"Iteration limit enabled: only running first {len(rolling_splits)} split(s)."
        )

    if USE_MDGNN:
        max_supported_splits = len(normalize_snapshot_file_list(GRAPH_SPLIT_FILES)) - 1
        print(
            f"Graph-based iterations available from provided files: {max_supported_splits}"
        )
        if len(rolling_splits) > max_supported_splits:
            print(
                f"Warning: {len(rolling_splits)} data splits were created, but only {max_supported_splits} graph-based "
                f"iterations are supported by the provided graph files. Extra data splits will be skipped."
            )

    loaded_graph_caches = {}

    for split_idx, split_info in enumerate(rolling_splits, start=1):
        print(f"Starting split {split_idx}...")

        train_graph_cache = EmptyGraphFeatureCache(hidden_dim=HIDDEN_DIM)
        eval_graph_cache = EmptyGraphFeatureCache(hidden_dim=HIDDEN_DIM)

        if USE_MDGNN:
            try:
                train_graph_file, eval_graph_file = get_split_graph_files(
                    split_idx, GRAPH_SPLIT_FILES
                )
            except IndexError:
                print(
                    f"Split {split_idx} does not have enough graph files available. Skipping."
                )
                continue

            required_files = {train_graph_file, eval_graph_file}

            stale_files = [f for f in loaded_graph_caches if f not in required_files]
            for stale_file in stale_files:
                del loaded_graph_caches[stale_file]
            if len(stale_files) > 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if train_graph_file not in loaded_graph_caches:
                loaded_graph_caches[train_graph_file] = SnapshotGraphFeatureCache(
                    snapshot_files=train_graph_file,
                    mdgnn_model=mdgnn,
                    device=device,
                    hidden_dim=HIDDEN_DIM,
                )

            if eval_graph_file not in loaded_graph_caches:
                loaded_graph_caches[eval_graph_file] = SnapshotGraphFeatureCache(
                    snapshot_files=eval_graph_file,
                    mdgnn_model=mdgnn,
                    device=device,
                    hidden_dim=HIDDEN_DIM,
                )

            train_graph_cache = loaded_graph_caches[train_graph_file]
            eval_graph_cache = loaded_graph_caches[eval_graph_file]

            print(
                f"Split {split_idx} graph files | "
                f"train={train_graph_file} | val/test={eval_graph_file} | "
                f"loaded_now={list(loaded_graph_caches.keys())}"
            )

        train_loader, val_loader, test_loader = build_loaders_for_split(
            split_dict=split_info,
            stock2id=stock2id,
            train_graph_cache=train_graph_cache,
            eval_graph_cache=eval_graph_cache,
            batch_size=BATCH_SIZE,
            max_headline_len=lhv.max_headline_len,
            feature_cols=available_numeric_features,
            target_col=TARGET_COL,
            window_size=WINDOW_SIZE,
            use_graph=USE_MDGNN,
        )

        if (
            len(train_loader.dataset) == 0
            or len(val_loader.dataset) == 0
            or len(test_loader.dataset) == 0
        ):
            print(f"Split {split_idx} is empty. Skipping.")
            continue

        lstm_encoder = None
        if USE_LSTM:
            lstm_encoder = LSTM_Encoder(
                vocab_size=len(lhv.word2id),
                embedding_dim=lhv.vector_size,
                hidden_dim=HIDDEN_DIM,
                embedding_matrix=lhv.embedding_matrix,
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
        save_path = f"data/model/output/best_{experiment_name}_split_{split_idx}.pt"

        epoch_pbar = tqdm(
            range(MAX_EPOCHS), desc=f"Split {split_idx} Epochs", unit="epoch"
        )

        for epoch in epoch_pbar:
            model.train()
            train_loss = 0.0

            # Wrap the DataLoader for batch progress
            train_pbar = tqdm(
                train_loader, desc="  Training", leave=False, unit="batch"
            )
            for (
                X_text_batch,
                X_num_batch,
                Y_batch,
                stock_batch,
                graph_seq_batch,
            ) in train_pbar:
                X_text_batch = X_text_batch.to(device)
                X_num_batch = X_num_batch.to(device)
                Y_batch = Y_batch.to(device)
                stock_batch = stock_batch.to(device)
                graph_seq_batch = graph_seq_batch.to(device)

                if (
                    torch.isnan(X_text_batch).any()
                    or torch.isnan(X_num_batch).any()
                    or torch.isnan(Y_batch).any()
                    or torch.isnan(graph_seq_batch).any()
                ):
                    print(
                        f"NaNs found in inputs! Text: {torch.isnan(X_text_batch).any()}, Num: {torch.isnan(X_num_batch).any()}, Y: {torch.isnan(Y_batch).any()}, Graph: {torch.isnan(graph_seq_batch).any()}"
                    )
                    import sys

                    sys.exit(1)

                optimizer.zero_grad()
                preds = model(
                    text_ids=X_text_batch,
                    numeric_feats=X_num_batch,
                    stock_ids=stock_batch,
                    graph_seq=graph_seq_batch,
                )

                if torch.isnan(preds).any():
                    print("NaNs found in predictions!")
                    import sys

                    sys.exit(1)

                loss = criterion(preds, Y_batch)
                loss.backward()
                optimizer.step()

                current_loss = loss.item()
                train_loss += current_loss
                train_pbar.set_postfix(loss=f"{current_loss:.4f}")

            model.eval()
            val_loss = 0.0
            val_pbar = tqdm(val_loader, desc="  Validating", leave=False, unit="batch")
            with torch.no_grad():
                for X_text_val, X_num_val, Y_val, stock_val, graph_seq_val in val_pbar:
                    X_text_val, X_num_val, Y_val, stock_val, graph_seq_val = [
                        t.to(device)
                        for t in [
                            X_text_val,
                            X_num_val,
                            Y_val,
                            stock_val,
                            graph_seq_val,
                        ]
                    ]

                    preds = model(
                        text_ids=X_text_val,
                        numeric_feats=X_num_val,
                        stock_ids=stock_val,
                        graph_seq=graph_seq_val,
                    )
                    val_loss += criterion(preds, Y_val).item()

            avg_train_loss = train_loss / max(len(train_loader), 1)
            avg_val_loss = val_loss / max(len(val_loader), 1)

            epoch_pbar.set_postfix(
                train=f"{avg_train_loss:.4f}", val=f"{avg_val_loss:.4f}"
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
                torch.save(model.state_dict(), save_path)
            else:
                counter += 1
                if counter >= PATIENCE:
                    tqdm.write(f"Early stopping triggered at epoch {epoch+1}")
                    break

        model.load_state_dict(torch.load(save_path, map_location=device))
        model.eval()

        test_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for (
                X_text_test,
                X_num_test,
                Y_test,
                stock_test,
                graph_seq_test,
            ) in test_loader:
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
                all_targets.extend(Y_test.detach().cpu().tolist())

        avg_test_loss = test_loss / max(len(test_loader), 1)
        print(f"{experiment_name} | Split {split_idx} | Test={avg_test_loss:.6f}")

        # Save predictions and targets to CSV
        output_dir = Path("data/model/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        results_df = pd.DataFrame(
            [
                {
                    "Date": row.get("Date"),
                    "Stock_symbol": row.get("Stock_symbol"),
                    "target_return": target,
                    "prediction": pred,
                }
                for row, target, pred in zip(
                    test_loader.dataset.valid_rows, all_targets, all_preds
                )
            ]
        )
        csv_path = output_dir / f"preds_{experiment_name}_split_{split_idx}.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"Predictions saved to: {csv_path}")

        del train_loader, val_loader, test_loader, model
        if USE_LSTM and lstm_encoder is not None:
            del lstm_encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
