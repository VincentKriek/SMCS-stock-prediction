import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_segment_softmax(scores: torch.Tensor, index: torch.Tensor, num_segments: int):
    out = torch.zeros_like(scores)
    for i in range(num_segments):
        mask = (index == i)
        if mask.any():
            out[mask] = F.softmax(scores[mask], dim=0)
    return out


class RelationGraphAttention(nn.Module):
    
    #Relation-specific graph attention layer, supports multi-head attention and mean pooling over heads.
    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int = 0,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.src_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dst_proj = nn.Linear(hidden_dim, hidden_dim)

        if edge_dim > 0:
            self.edge_proj = nn.Linear(edge_dim, hidden_dim)
            attn_in_dim = self.head_dim * 3
        else:
            self.edge_proj = None
            attn_in_dim = self.head_dim * 2

        self.attn_fc = nn.Linear(attn_in_dim, 1)
        self.msg_fc = nn.Linear(self.head_dim, self.head_dim)

        self.merge_fc = nn.Linear(self.head_dim, hidden_dim)
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
        num_edges = src_idx.size(0)

        if num_edges == 0:
            return dst_x

        src_feat = self.src_proj(src_x[src_idx]).view(num_edges, self.num_heads, self.head_dim)
        dst_feat = self.dst_proj(dst_x[dst_idx]).view(num_edges, self.num_heads, self.head_dim)

        if self.edge_proj is not None and edge_attr is not None:
            e_feat = self.edge_proj(edge_attr).view(num_edges, self.num_heads, self.head_dim)
            attn_in = torch.cat([src_feat, dst_feat, e_feat], dim=-1)
            msg = self.msg_fc(src_feat + e_feat)
        else:
            attn_in = torch.cat([src_feat, dst_feat], dim=-1)
            msg = self.msg_fc(src_feat)

        attn_scores = self.attn_fc(torch.tanh(attn_in)).squeeze(-1)  # [E, H]

        alpha_heads = []
        for h in range(self.num_heads):
            alpha_h = masked_segment_softmax(attn_scores[:, h], dst_idx, num_dst)
            alpha_heads.append(alpha_h)
        alpha = torch.stack(alpha_heads, dim=1)  # [E, H]

        weighted_msg = msg * alpha.unsqueeze(-1)  # [E, H, Dh]

        agg = torch.zeros(num_dst, self.num_heads, self.head_dim, device=device)
        for h in range(self.num_heads):
            agg[:, h, :].index_add_(0, dst_idx, weighted_msg[:, h, :])

        # average pooling over heads
        agg = agg.mean(dim=1)  # [N_dst, Dh]

        agg = self.merge_fc(agg)  # [N_dst, D]

        updated = self.out_fc(torch.cat([dst_x, agg], dim=-1))
        updated = self.dropout(updated)
        updated = self.norm(dst_x + updated)

        return updated


class MetaPathAggregator(nn.Module):
    
    # Aggregate multiple path-based node representations.
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
        assert len(path_reprs) > 0, "path_reprs must not be empty"

        stacked = torch.stack(path_reprs, dim=1)     # [N, P, D]
        h = torch.tanh(self.path_fc(stacked))        # [N, P, D]
        scores = self.score_fc(h).squeeze(-1)        # [N, P]
        alpha = F.softmax(scores, dim=1)             # [N, P]

        out = torch.sum(alpha.unsqueeze(-1) * stacked, dim=1)  # [N, D]
        out = self.out_fc(out)
        out = self.dropout(out)
        out = self.norm(out)

        return out


class IntraDaySnapshotEncoder(nn.Module):

    # Encode one graph snapshot and return stock embeddings.
    def __init__(
        self,
        stock_in_dim: int,
        bank_in_dim: int,
        industry_in_dim: int,
        edge_dims: Dict[str, int],
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
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

        # Only used if industry features are provided
        self.industry_encoder = nn.Sequential(
            nn.Linear(industry_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                "SS": RelationGraphAttention(
                    hidden_dim=hidden_dim,
                    edge_dim=edge_dims.get("SS", 0),
                    num_heads=num_heads,
                    dropout=dropout
                ),
                "SB": RelationGraphAttention(
                    hidden_dim=hidden_dim,
                    edge_dim=edge_dims.get("SB", 0),
                    num_heads=num_heads,
                    dropout=dropout
                ),
                "BS": RelationGraphAttention(
                    hidden_dim=hidden_dim,
                    edge_dim=edge_dims.get("BS", edge_dims.get("SB", 0)),
                    num_heads=num_heads,
                    dropout=dropout
                ),
                "SI": RelationGraphAttention(
                    hidden_dim=hidden_dim,
                    edge_dim=edge_dims.get("SI", 0),
                    num_heads=num_heads,
                    dropout=dropout
                ),
                "IS": RelationGraphAttention(
                    hidden_dim=hidden_dim,
                    edge_dim=edge_dims.get("IS", edge_dims.get("SI", 0)),
                    num_heads=num_heads,
                    dropout=dropout
                ),
                "II": RelationGraphAttention(
                    hidden_dim=hidden_dim,
                    edge_dim=edge_dims.get("II", 0),
                    num_heads=num_heads,
                    dropout=dropout
                ),
            })
            self.layers.append(layer)

        self.meta_agg = MetaPathAggregator(hidden_dim, num_paths=3, dropout=dropout)

    def forward(
        self,
        stock_feat: torch.Tensor,
        bank_feat: Optional[torch.Tensor],
        industry_feat: Optional[torch.Tensor],
        edges: Dict[str, Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]]]
    ) -> torch.Tensor:
        
        # Return stock node embeddings after message passing.
        stock_h = self.stock_encoder(stock_feat)

        bank_h = None
        if bank_feat is not None:
            bank_h = self.bank_encoder(bank_feat)

        industry_h = None
        if industry_feat is not None:
            industry_h = self.industry_encoder(industry_feat)

        for layer in self.layers:
            stock_paths = [stock_h]

            # SS: stock -> stock
            if edges.get("SS") is not None:
                edge_index_ss, edge_attr_ss = edges["SS"]
                stock_from_stock = layer["SS"](stock_h, stock_h, edge_index_ss, edge_attr_ss)
                stock_paths.append(stock_from_stock)

            # SB: bank -> stock
            if bank_h is not None and edges.get("SB") is not None:
                edge_index_sb, edge_attr_sb = edges["SB"]
                stock_from_bank = layer["SB"](bank_h, stock_h, edge_index_sb, edge_attr_sb)
                stock_paths.append(stock_from_bank)

            # SI: industry -> stock
            if industry_h is not None and edges.get("SI") is not None:
                edge_index_si, edge_attr_si = edges["SI"]
                stock_from_industry = layer["SI"](industry_h, stock_h, edge_index_si, edge_attr_si)
                stock_paths.append(stock_from_industry)

            stock_h = self.meta_agg(stock_paths)

            # Update bank nodes through BS: stock -> bank
            if bank_h is not None and edges.get("BS") is not None:
                edge_index_bs, edge_attr_bs = edges["BS"]
                bank_h = layer["BS"](stock_h, bank_h, edge_index_bs, edge_attr_bs)

            # Update industry nodes if provided
            if industry_h is not None and edges.get("IS") is not None:
                edge_index_is, edge_attr_is = edges["IS"]
                industry_h = layer["IS"](stock_h, industry_h, edge_index_is, edge_attr_is)

            if industry_h is not None and edges.get("II") is not None:
                edge_index_ii, edge_attr_ii = edges["II"]
                industry_h = layer["II"](industry_h, industry_h, edge_index_ii, edge_attr_ii)

        return stock_h


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


class MDGNN(nn.Module):
    # MDGNN:    graph snapshot encoder -> temporal layer -> prediction head
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
            num_heads=num_heads,
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

    def forward(self, x_seq: torch.Tensor, return_attention: bool = False):
        return self.forward_from_sequence(x_seq, return_attention=return_attention)
