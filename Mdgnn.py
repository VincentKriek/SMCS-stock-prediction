import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F



def masked_segment_softmax(scores: torch.Tensor, index: torch.Tensor, num_segments: int):
    """
    Softmax over variable-size segments.
    scores: [E]
    index:  [E] target node indices
    """
    out = torch.zeros_like(scores)
    for i in range(num_segments):
        mask = (index == i)
        if mask.any():
            out[mask] = F.softmax(scores[mask], dim=0)
    return out



# Relation-specific graph attention
class RelationGraphAttention(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int = 0, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim

        self.src_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dst_proj = nn.Linear(hidden_dim, hidden_dim)

        if edge_dim > 0:
            self.edge_proj = nn.Linear(edge_dim, hidden_dim)
            attn_in_dim = hidden_dim * 3
        else:
            self.edge_proj = None
            attn_in_dim = hidden_dim * 2

        self.attn_fc = nn.Linear(attn_in_dim, 1)
        self.msg_fc = nn.Linear(hidden_dim, hidden_dim)
        self.out_fc = nn.Linear(hidden_dim * 2, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        src_x: torch.Tensor,
        dst_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        device = dst_x.device
        src_idx = edge_index[0]
        dst_idx = edge_index[1]
        num_dst = dst_x.size(0)

        src_feat = self.src_proj(src_x[src_idx])     # [E, D]
        dst_feat = self.dst_proj(dst_x[dst_idx])     # [E, D]

        if self.edge_proj is not None and edge_attr is not None:
            e_feat = self.edge_proj(edge_attr)       # [E, D]
            attn_in = torch.cat([src_feat, dst_feat, e_feat], dim=-1)
            msg = self.msg_fc(src_feat + e_feat)
        else:
            attn_in = torch.cat([src_feat, dst_feat], dim=-1)
            msg = self.msg_fc(src_feat)

        attn_scores = self.attn_fc(torch.tanh(attn_in)).squeeze(-1)  # [E]
        alpha = masked_segment_softmax(attn_scores, dst_idx, num_dst)  # [E]

        weighted_msg = msg * alpha.unsqueeze(-1)  # [E, D]

        agg = torch.zeros(num_dst, self.hidden_dim, device=device)
        agg.index_add_(0, dst_idx, weighted_msg)

        updated = self.out_fc(torch.cat([dst_x, agg], dim=-1))
        updated = self.dropout(updated)
        updated = self.norm(dst_x + updated)

        return updated


# Meta-path aggregation
class MetaPathAggregator(nn.Module):
    def __init__(self, hidden_dim: int, num_paths: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_paths = num_paths

        self.path_fc = nn.Linear(hidden_dim, hidden_dim)
        self.score_fc = nn.Linear(hidden_dim, 1, bias=False)

        self.out_fc = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, path_reprs: List[torch.Tensor]) -> torch.Tensor:
        """
        path_reprs: list of [N, D]
        """
        assert len(path_reprs) > 0, "path_reprs must not be empty"
        stacked = torch.stack(path_reprs, dim=1)  # [N, P, D]

        h = torch.tanh(self.path_fc(stacked))      # [N, P, D]
        scores = self.score_fc(h).squeeze(-1)      # [N, P]
        alpha = F.softmax(scores, dim=1)           # [N, P]

        out = torch.sum(alpha.unsqueeze(-1) * stacked, dim=1)  # [N, D]
        out = self.out_fc(out)
        out = self.dropout(out)
        out = self.norm(out)

        return out


# Intra-day graph snapshot encoder
class IntraDaySnapshotEncoder(nn.Module):
    def __init__(
        self,
        stock_in_dim: int,
        bank_in_dim: int,
        industry_in_dim: int,
        edge_dims: Dict[str, int],
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.stock_encoder = nn.Sequential(
            nn.Linear(stock_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.bank_encoder = nn.Sequential(
            nn.Linear(bank_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.industry_encoder = nn.Sequential(
            nn.Linear(industry_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                "SS": RelationGraphAttention(hidden_dim, edge_dims.get("SS", 0), dropout),
                "SB": RelationGraphAttention(hidden_dim, edge_dims.get("SB", 0), dropout),
                "BS": RelationGraphAttention(hidden_dim, edge_dims.get("BS", edge_dims.get("SB", 0)), dropout),
                "SI": RelationGraphAttention(hidden_dim, edge_dims.get("SI", 0), dropout),
                "IS": RelationGraphAttention(hidden_dim, edge_dims.get("IS", edge_dims.get("SI", 0)), dropout),
                "II": RelationGraphAttention(hidden_dim, edge_dims.get("II", 0), dropout),
            })
            self.layers.append(layer)

        # aggregate stock views from distinct relations/meta-paths
        self.meta_agg = MetaPathAggregator(hidden_dim, num_paths=3, dropout=dropout)

    def forward(
        self,
        stock_feat: torch.Tensor,
        bank_feat: Optional[torch.Tensor],
        industry_feat: Optional[torch.Tensor],
        edges: Dict[str, Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]]]
    ) -> torch.Tensor:
        stock_x = self.stock_encoder(stock_feat)

        if bank_feat is not None and bank_feat.numel() > 0:
            bank_x = self.bank_encoder(bank_feat)
        else:
            bank_x = None

        if industry_feat is not None and industry_feat.numel() > 0:
            industry_x = self.industry_encoder(industry_feat)
        else:
            industry_x = None

        for layer in self.layers:
            # relation-specific updates
            h_ss = stock_x
            h_sb = stock_x
            h_si = stock_x

            if edges.get("SS") is not None:
                ei, ea = edges["SS"]
                h_ss = layer["SS"](stock_x, stock_x, ei, ea)

            if bank_x is not None and edges.get("SB") is not None:
                ei, ea = edges["SB"]
                h_sb = layer["SB"](bank_x, stock_x, ei, ea)

            if industry_x is not None and edges.get("SI") is not None:
                ei, ea = edges["SI"]
                h_si = layer["SI"](industry_x, stock_x, ei, ea)

            # meta-path aggregation over stock representations
            stock_x = self.meta_agg([h_ss, h_sb, h_si])

            # optionally update bank / industry 
            if bank_x is not None and edges.get("BS") is not None:
                ei, ea = edges["BS"]
                bank_x = layer["BS"](stock_x, bank_x, ei, ea)

            if industry_x is not None and edges.get("IS") is not None:
                ei, ea = edges["IS"]
                industry_x = layer["IS"](stock_x, industry_x, ei, ea)

            if industry_x is not None and edges.get("II") is not None:
                ei, ea = edges["II"]
                industry_x = layer["II"](industry_x, industry_x, ei, ea)

        return stock_x   # [Ns, D]


# Inter-day temporal extraction layer
class ALiBiSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1, alibi_slope: float = 1.0):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.alibi_slope = alibi_slope

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _build_causal_mask(self, T: int, device):
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask.unsqueeze(0).unsqueeze(0)

    def _build_alibi_bias(self, T: int, device):
        pos = torch.arange(T, device=device)
        rel_dist = pos.unsqueeze(1) - pos.unsqueeze(0)
        rel_dist = rel_dist.clamp(min=0).float()
        bias = -self.alibi_slope * rel_dist
        return bias.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        B, T, D = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        scores = scores + self._build_causal_mask(T, x.device) + self._build_alibi_bias(T, x.device)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)
        return out, attn


class TemporalExtractionLayer(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=4, ff_dim=256, dropout=0.1, alibi_slope=1.0):
        super().__init__()
        self.attn = ALiBiSelfAttention(hidden_dim, num_heads, dropout, alibi_slope)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
        )

    def forward(self, x):
        attn_out, attn_weights = self.attn(x)
        x = self.norm1(x + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        final_repr = x[:, -1, :]
        return x, final_repr, attn_weights



# Prediction
class PredictionHead(nn.Module):
    def __init__(self, hidden_dim=128, task="regression", dropout=0.1):
        super().__init__()
        assert task in ["regression", "classification"]
        self.task = task
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        logits = self.mlp(x).squeeze(-1)
        if self.task == "classification":
            return torch.sigmoid(logits)
        return logits


# MDGNN: per-day graph snapshot -> stock embeddings -> temporal layer -> prediction
class MDGNN(nn.Module):
    def __init__(
        self,
        stock_in_dim: int,
        bank_in_dim: int,
        industry_in_dim: int,
        edge_dims: Dict[str, int],
        hidden_dim: int = 128,
        gnn_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,
        alibi_slope: float = 1.0,
        task: str = "regression"
    ):
        super().__init__()

        self.snapshot_encoder = IntraDaySnapshotEncoder(
            stock_in_dim=stock_in_dim,
            bank_in_dim=bank_in_dim,
            industry_in_dim=industry_in_dim,
            edge_dims=edge_dims,
            hidden_dim=hidden_dim,
            num_layers=gnn_layers,
            dropout=dropout
        )

        self.temporal_layer = TemporalExtractionLayer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            alibi_slope=alibi_slope
        )

        self.pred_head = PredictionHead(
            hidden_dim=hidden_dim,
            task=task,
            dropout=dropout
        )

    def encode_snapshot(
        self,
        stock_feat: torch.Tensor,
        bank_feat: Optional[torch.Tensor],
        industry_feat: Optional[torch.Tensor],
        edges: Dict[str, Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]]]
    ):
        return self.snapshot_encoder(stock_feat, bank_feat, industry_feat, edges)

    def forward_from_sequence(self, x_seq: torch.Tensor, return_attention: bool = False):
        _, final_repr, attn = self.temporal_layer(x_seq)
        pred = self.pred_head(final_repr)
        if return_attention:
            return pred, final_repr, attn
        return pred