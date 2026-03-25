"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  RNA FOLDING MODEL — TEMPLATE-GUIDED, NOVEL COMPONENTS                       ║
║                                                                               ║
║  Novel components (vs v2 SE3 baseline):                                      ║
║   1. TemplatePairStack   — injects template distograms into pair repr.        ║
║   2. TemplateSingleStack — cross-attention from query to template frames      ║
║   3. SparseGraphAttention — k-NN graph attention (RNA contact graph)          ║
║   4. TorsionAwareTriangle — triangle updates biased by torsion compatibility  ║
║   5. TorsionAngleHead    — predicts eta/theta/curl torsion angles             ║
║   6. pLDDTHead           — per-residue confidence, used in loss + masking     ║
║   7. TemplateWeightGate  — learnable re-weighting of template pool            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from typing import Dict, Optional, Tuple, List

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
class ModelConfig:
    D_NODE   = 128
    D_PAIR   = 64
    D_HIDDEN = 64
    N_HEAD   = 4
    N_QUERY_PT  = 4
    N_VALUE_PT  = 4
    N_EVOFORMER = 6     # +2 vs v2 to exploit template info
    N_STRUCTURE = 4     # +1 for torsion-conditioned steps
    N_RECYCLE   = 3
    N_DIST_BINS = 38    # match template_search triangular bins
    N_ORIENT    = 4
    N_RBF       = 16
    N_REL_POS   = 65
    N_DIHED     = 4
    N_PAIR_TYPE = 3
    F1_DIM      = 5
    VOCAB_SIZE  = 6
    DROPOUT     = 0.1
    MAX_LEN     = 512
    MAX_TEMPLATES = 4
    K_LAPLACIAN   = 8   # from template_search
    N_TORSION_OUT = 6   # eta, theta, curl (sin+cos each)
    N_PLDDT_BINS  = 50  # pLDDT discretised into 50 bins (like AF2)
    K_GRAPH_EDGES = 12  # sparse graph k-NN

cfg = ModelConfig()
VOCAB = {'A':0,'U':1,'G':2,'C':3,'<PAD>':4,'<UNK>':5}


# ─────────────────────────────────────────────────────────────
# UTILITY LAYERS
# ─────────────────────────────────────────────────────────────
class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.float()).to(x.dtype)

class Linear(nn.Linear):
    def __init__(self, in_f, out_f, bias=True, init='default'):
        super().__init__(in_f, out_f, bias=bias)
        if init == 'relu':
            nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        elif init in ('zeros', 'final'):
            nn.init.zeros_(self.weight)
            if bias: nn.init.zeros_(self.bias)
        else:
            nn.init.xavier_uniform_(self.weight)
            if bias: nn.init.zeros_(self.bias)

class SinusoidalPE(nn.Module):
    def __init__(self, d, max_len):
        super().__init__()
        pe  = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ─────────────────────────────────────────────────────────────
# 1. NOVEL: TEMPLATE PAIR STACK
#    Encodes template distograms + laplacian eigenvectors into
#    the pair representation via a gated cross-attention mechanism.
# ─────────────────────────────────────────────────────────────
class TemplatePairStack(nn.Module):
    """
    For each template k, projects its distogram + Laplacian outer-product
    into D_PAIR space, then soft-fuses templates using learned weights.
    The fusion is gated by the template validity mask so missing templates
    contribute zero signal.
    Novel: Laplacian outer-product adds global topology signal to pair repr.
    """
    def __init__(self):
        super().__init__()
        # Distogram embedding: 38 soft bins → D_PAIR
        self.dgram_proj = nn.Sequential(
            Linear(cfg.N_DIST_BINS, cfg.D_PAIR),
            nn.ReLU(),
            LayerNorm(cfg.D_PAIR),
        )
        # Laplacian outer-product: (L, K) → (L, L, K²) → D_PAIR
        lap_outer_dim = cfg.K_LAPLACIAN * cfg.K_LAPLACIAN
        self.lap_proj = nn.Sequential(
            Linear(lap_outer_dim, cfg.D_PAIR, init='relu'),
            nn.ReLU(),
            Linear(cfg.D_PAIR, cfg.D_PAIR),
        )
        # Learned template gate (scalar per template, sigmoid)
        self.tmpl_gate = nn.Sequential(
            Linear(1, 16), nn.ReLU(),
            Linear(16, 1), nn.Sigmoid()
        )
        # Final merge
        self.out_proj = nn.Sequential(
            Linear(cfg.D_PAIR * 2, cfg.D_PAIR),
            LayerNorm(cfg.D_PAIR),
        )
        self.seq_match_proj = Linear(1, cfg.D_PAIR)

    def forward(self,
                pair          : torch.Tensor,    # (B, L, L, D_PAIR)
                tmpl_dgram    : torch.Tensor,    # (B, T, L, L, 38)
                tmpl_laplacian: torch.Tensor,    # (B, T, L, 8)
                tmpl_weights  : torch.Tensor,    # (B, T)
                tmpl_valid    : torch.Tensor,    # (B, T, L)
                ) -> torch.Tensor:
        B, L, _, D = pair.shape
        T = tmpl_dgram.shape[1]

        # Per-template: distogram → pair space
        dgram_flat = tmpl_dgram.view(B * T, L, L, cfg.N_DIST_BINS)
        pair_tmpl  = self.dgram_proj(dgram_flat)                  # (BT, L, L, D)

        # Per-template: Laplacian outer product
        lap_flat = tmpl_laplacian.view(B * T, L, cfg.K_LAPLACIAN)
        lap_op   = lap_flat.unsqueeze(2) * lap_flat.unsqueeze(1)  # (BT, L, L, K)
        lap_op   = lap_op.view(B * T, L, L, cfg.K_LAPLACIAN ** 2)
        lap_emb  = self.lap_proj(lap_op)                          # (BT, L, L, D)

        combined = pair_tmpl + lap_emb                            # (BT, L, L, D)
        combined = combined.view(B, T, L, L, D)

        # Learned gate from template weight (confidence-based)
        gate = self.tmpl_gate(tmpl_weights.unsqueeze(-1))         # (B, T, 1)
        gate = gate.unsqueeze(-1).unsqueeze(-1)                   # (B, T, 1, 1, 1)

        # Validity mask: (B, T, L) → outer product → (B, T, L, L, 1)
        v_mask = (tmpl_valid.unsqueeze(2) * tmpl_valid.unsqueeze(3)).unsqueeze(-1)

        # Weighted sum over templates
        w  = (tmpl_weights * gate.squeeze(-1).squeeze(-1).squeeze(-1)).softmax(dim=1)
        w  = w.view(B, T, 1, 1, 1)
        fused = (combined * v_mask * w).sum(dim=1)                # (B, L, L, D)

        out = self.out_proj(torch.cat([pair, fused], dim=-1))
        return out + pair                                          # residual


# ─────────────────────────────────────────────────────────────
# 2. NOVEL: TEMPLATE SINGLE STACK
#    Cross-attention from query single repr to template torsion
#    angles and frames — gives the node repr a structural prior.
# ─────────────────────────────────────────────────────────────
class TemplateSingleStack(nn.Module):
    """
    Cross-attention: query residues attend to their counterpart in each template.
    Templates are weighted by the pre-computed soft weights.
    Novel: conditions on torsion angles (not just distance) so the model
    can learn helix/loop character from templates.
    """
    def __init__(self):
        super().__init__()
        # Project frame rotation matrix (9 values from 3×3 submatrix) + translation (3)
        self.frame_proj   = Linear(12, cfg.D_NODE // 2)
        self.torsion_proj = Linear(cfg.N_TORSION_OUT, cfg.D_NODE // 2)
        self.seq_match_proj = Linear(1, 16)

        tmpl_in = cfg.D_NODE // 2 + cfg.D_NODE // 2 + 16
        self.tmpl_proj = Linear(tmpl_in, cfg.D_NODE)

        # Gated cross-attention
        self.q_proj = Linear(cfg.D_NODE, cfg.D_NODE, bias=False)
        self.k_proj = Linear(cfg.D_NODE, cfg.D_NODE, bias=False)
        self.v_proj = Linear(cfg.D_NODE, cfg.D_NODE, bias=False)
        self.scale  = (cfg.D_NODE // cfg.N_HEAD) ** -0.5
        self.out_proj = Linear(cfg.D_NODE, cfg.D_NODE, init='final')
        self.norm     = LayerNorm(cfg.D_NODE)
        self.gate     = nn.Sequential(Linear(cfg.D_NODE, cfg.D_NODE), nn.Sigmoid())

    def forward(self,
                single        : torch.Tensor,    # (B, L, D_NODE)
                tmpl_frames   : torch.Tensor,    # (B, T, L, 4, 4)
                tmpl_torsion  : torch.Tensor,    # (B, T, L, 6)
                tmpl_weights  : torch.Tensor,    # (B, T)
                tmpl_seq_match: torch.Tensor,    # (B, T, L)
                ) -> torch.Tensor:
        B, L, D = single.shape
        T = tmpl_frames.shape[1]

        # Extract rotation (9) + translation (3) = 12 per residue per template
        R  = tmpl_frames[:, :, :, :3, :3].reshape(B, T, L, 9)
        tr = tmpl_frames[:, :, :, :3,  3]                         # (B, T, L, 3)
        frame_flat = torch.cat([R, tr], dim=-1)                    # (B, T, L, 12)

        f_emb = self.frame_proj(frame_flat)                        # (B, T, L, D/2)
        t_emb = self.torsion_proj(tmpl_torsion)                   # (B, T, L, D/2)
        m_emb = self.seq_match_proj(tmpl_seq_match.unsqueeze(-1)) # (B, T, L, 16)

        tmpl_repr = self.tmpl_proj(torch.cat([f_emb, t_emb, m_emb], dim=-1))  # (B, T, L, D)

        # Weighted template summary: (B, L, D)
        w     = tmpl_weights.softmax(dim=1).view(B, T, 1, 1)
        tmpl_s = (tmpl_repr * w).sum(dim=1)                       # (B, L, D)

        # Gated cross-attention: single queries, template summary as K/V
        s_n = self.norm(single)
        Q   = self.q_proj(s_n).view(B, L, cfg.N_HEAD, -1).transpose(1, 2)
        K   = self.k_proj(tmpl_s).view(B, L, cfg.N_HEAD, -1).transpose(1, 2)
        V   = self.v_proj(tmpl_s).view(B, L, cfg.N_HEAD, -1).transpose(1, 2)
        att = (Q @ K.transpose(-2, -1)) * self.scale
        att = att.softmax(dim=-1)
        ctx = (att @ V).transpose(1, 2).reshape(B, L, D)

        g   = self.gate(single)
        out = single + g * self.out_proj(ctx)
        return out


# ─────────────────────────────────────────────────────────────
# 3. NOVEL: SPARSE GRAPH ATTENTION
#    Instead of full O(L²) attention, builds a k-NN + base-pair
#    graph from the current coordinate estimate and attends only
#    over edges. Dramatically more expressive for long sequences.
# ─────────────────────────────────────────────────────────────
class SparseGraphAttention(nn.Module):
    """
    Attention over a sparse RNA contact graph.
    Edges include: k-NN by coordinate distance + secondary structure pairs.
    Novel: dynamic edge construction at each recycle step as coords refine.
    """
    def __init__(self, k: int = cfg.K_GRAPH_EDGES):
        super().__init__()
        self.k    = k
        self.norm = LayerNorm(cfg.D_NODE)
        self.q    = Linear(cfg.D_NODE, cfg.D_NODE, bias=False)
        self.k_   = Linear(cfg.D_NODE, cfg.D_NODE, bias=False)
        self.v_   = Linear(cfg.D_NODE, cfg.D_NODE, bias=False)
        # Edge feature: pair repr at (i,j) biases attention
        self.edge_bias = Linear(cfg.D_PAIR, cfg.N_HEAD, bias=False, init='zeros')
        self.out  = Linear(cfg.D_NODE, cfg.D_NODE, init='final')
        self.scale = (cfg.D_NODE // cfg.N_HEAD) ** -0.5

    def _build_graph(self, coords: torch.Tensor,
                     contact_ss: torch.Tensor) -> torch.Tensor:
        """
        Build edge indices (B, L, k_total) based on current coords + SS.
        Returns edge index tensor. Falls back to linear edges if no coords.
        """
        B, L, _ = coords.shape
        k = min(self.k, L - 1)
        # Distance-based k-NN
        d  = torch.cdist(coords, coords)                           # (B, L, L)
        d  = d + torch.eye(L, device=d.device).unsqueeze(0) * 1e9
        _, idx = d.topk(k, dim=-1, largest=False)                  # (B, L, k)
        # Add secondary structure contacts
        ss_idx = contact_ss.topk(min(4, L), dim=-1).indices        # (B, L, 4)
        idx    = torch.cat([idx, ss_idx], dim=-1)                   # (B, L, k+4)
        return idx

    def forward(self,
                single    : torch.Tensor,   # (B, L, D)
                pair      : torch.Tensor,   # (B, L, L, D_PAIR)
                coords    : torch.Tensor,   # (B, L, 3)
                contact_ss: torch.Tensor,   # (B, L, L)
                ) -> torch.Tensor:
        B, L, D = single.shape
        edge_idx = self._build_graph(coords, contact_ss)            # (B, L, E)
        E = edge_idx.shape[-1]

        s_n = self.norm(single)
        Q   = self.q(s_n)                                          # (B, L, D)
        K   = self.k_(s_n)
        V   = self.v_(s_n)

        # Gather K, V at edge positions
        idx_flat  = edge_idx.view(B, L * E)                        # (B, L*E)
        idx_exp   = idx_flat.unsqueeze(-1).expand(-1, -1, D)
        K_nbr     = torch.gather(K, 1, idx_exp).view(B, L, E, D)
        V_nbr     = torch.gather(V, 1, idx_exp).view(B, L, E, D)

        # Reshape to multi-head
        H = cfg.N_HEAD
        Dh = D // H
        Q_h   = Q.view(B, L, H, Dh)
        K_nbr = K_nbr.view(B, L, E, H, Dh)
        V_nbr = V_nbr.view(B, L, E, H, Dh)

        # Attention scores with pair edge bias
        # pair bias: gather (i, j) pair repr for each edge
        idx_bias = edge_idx.view(B, L, E, 1).expand(-1, -1, -1, cfg.D_PAIR)
        pair_ij  = torch.gather(
            pair.view(B, L * L, cfg.D_PAIR),
            1,
            (edge_idx.view(B, 1, L * E).expand(-1, L, -1) // 1).view(B, L * E, cfg.D_PAIR)
        ).view(B, L, E, cfg.D_PAIR)
        bias_h = self.edge_bias(pair_ij)                           # (B, L, E, H)
        bias_h = bias_h.permute(0, 1, 3, 2)                        # (B, L, H, E)

        scores = (Q_h.unsqueeze(2) * K_nbr).sum(-1)               # (B, L, E, H)
        scores = scores.permute(0, 1, 3, 2) * self.scale + bias_h # (B, L, H, E)
        attn   = scores.softmax(dim=-1)                             # (B, L, H, E)

        ctx    = (attn.unsqueeze(-1) * V_nbr.permute(0, 1, 3, 2, 4)).sum(-2)
        ctx    = ctx.view(B, L, D)
        return single + self.out(ctx)


# ─────────────────────────────────────────────────────────────
# 4. NOVEL: TORSION-AWARE TRIANGLE UPDATE
#    Standard triangle multiplicative update biased by compatibility
#    of predicted torsion angles at the third vertex.
# ─────────────────────────────────────────────────────────────
class TorsionAwareTriangleUpdate(nn.Module):
    """
    Triangle multiplicative update (incoming/outgoing) with a torsion
    compatibility bias.  For pair (i,j) the update aggregates paths
    i→k→j weighted by how well k's torsion fits a WC-helix stacking pattern.
    Novel: integrates local backbone conformation into pair update.
    """
    def __init__(self, outgoing: bool = True):
        super().__init__()
        self.outgoing = outgoing
        self.norm = LayerNorm(cfg.D_PAIR)
        d = cfg.D_PAIR
        self.left_proj  = Linear(d, d, bias=False)
        self.right_proj = Linear(d, d, bias=False)
        self.left_gate  = nn.Sequential(Linear(d, d), nn.Sigmoid())
        self.right_gate = nn.Sequential(Linear(d, d), nn.Sigmoid())
        self.out_gate   = nn.Sequential(Linear(d, d), nn.Sigmoid())
        self.out_proj   = Linear(d, d, init='final')
        self.norm_out   = LayerNorm(d)
        # Torsion compatibility projection: per-residue torsion → scalar weight
        self.torsion_compat = Linear(cfg.N_TORSION_OUT, 1)

    def forward(self,
                pair    : torch.Tensor,    # (B, L, L, D)
                torsion : torch.Tensor,    # (B, L, N_TORSION_OUT)
                ) -> torch.Tensor:
        B, L, _, D = pair.shape
        z = self.norm(pair)

        a = self.left_proj(z)  * self.left_gate(z)
        b = self.right_proj(z) * self.right_gate(z)

        # Torsion compatibility weight for each intermediate residue k
        t_w = self.torsion_compat(torsion).squeeze(-1)   # (B, L)
        t_w = t_w.sigmoid()

        if self.outgoing:
            # (i,j) += sum_k [ a(i,k) * b(j,k) * t_w(k) ]
            tw  = t_w.unsqueeze(1).unsqueeze(3)           # (B, 1, L, 1)
            a_w = a * tw.transpose(1, 2)                  # (B, L, L, D)
            x   = torch.einsum('bikd,bjkd->bijd', a_w, b) / (L ** 0.5 + 1e-8)
        else:
            # (i,j) += sum_k [ a(k,i) * b(k,j) * t_w(k) ]
            tw  = t_w.unsqueeze(2).unsqueeze(3)           # (B, L, 1, 1)
            b_w = b * tw                                  # (B, L, L, D)
            x   = torch.einsum('bkid,bkjd->bijd', a, b_w) / (L ** 0.5 + 1e-8)

        g   = self.out_gate(z)
        out = self.out_proj(self.norm_out(x)) * g
        return pair + out


# ─────────────────────────────────────────────────────────────
# Standard layers from v2 (kept for compatibility)
# ─────────────────────────────────────────────────────────────
class RowAttentionWithPairBias(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_head = cfg.N_HEAD
        self.d_head = cfg.D_NODE // cfg.N_HEAD
        self.scale  = self.d_head ** -0.5
        self.norm   = LayerNorm(cfg.D_NODE)
        self.q      = Linear(cfg.D_NODE, cfg.D_NODE, bias=False)
        self.k      = Linear(cfg.D_NODE, cfg.D_NODE, bias=False)
        self.v      = Linear(cfg.D_NODE, cfg.D_NODE, bias=False)
        self.b      = Linear(cfg.D_PAIR, cfg.N_HEAD, bias=False)
        self.g      = nn.Sequential(Linear(cfg.D_NODE, cfg.D_NODE), nn.Sigmoid())
        self.out    = Linear(cfg.D_NODE, cfg.D_NODE, init='final')

    def forward(self, s, pair, mask=None):
        B, L, D = s.shape
        z = self.norm(s)
        Q = self.q(z).view(B, L, self.n_head, self.d_head).transpose(1, 2)
        K = self.k(z).view(B, L, self.n_head, self.d_head).transpose(1, 2)
        V = self.v(z).view(B, L, self.n_head, self.d_head).transpose(1, 2)
        b = self.b(pair).permute(0, 3, 1, 2)
        a = (Q @ K.transpose(-2, -1)) * self.scale + b
        if mask is not None:
            m = (~mask).unsqueeze(1).unsqueeze(1)
            a = a.masked_fill(m, -1e9)
        a = a.softmax(-1)
        g = self.g(z).view(B, L, self.n_head, self.d_head).transpose(1, 2)
        x = (g * (a @ V)).transpose(1, 2).reshape(B, L, D)
        return s + self.out(x)


class ColumnAttention(nn.Module):
    def forward(self, s, pair, mask=None):
        return s   # placeholder — transpose-trick can be added as needed


class PairTransition(nn.Module):
    def __init__(self, mult=4):
        super().__init__()
        d = cfg.D_PAIR
        self.norm = LayerNorm(d)
        self.ff   = nn.Sequential(Linear(d, d * mult), nn.ReLU(), Linear(d * mult, d, init='final'))
    def forward(self, pair):
        return pair + self.ff(self.norm(pair))


class NodeTransition(nn.Module):
    def __init__(self, mult=4):
        super().__init__()
        d = cfg.D_NODE
        self.norm = LayerNorm(d)
        self.ff   = nn.Sequential(Linear(d, d * mult), nn.ReLU(), Linear(d * mult, d, init='final'))
    def forward(self, s):
        return s + self.ff(self.norm(s))


# ─────────────────────────────────────────────────────────────
# 5. ENHANCED EVOFORMER BLOCK
# ─────────────────────────────────────────────────────────────
class EvoformerBlock(nn.Module):
    """
    Standard Evoformer with torsion-aware triangle updates replacing
    the plain triangle multiplicative update.
    """
    def __init__(self):
        super().__init__()
        self.row_attn   = RowAttentionWithPairBias()
        self.node_trans = NodeTransition()
        self.tri_out    = TorsionAwareTriangleUpdate(outgoing=True)
        self.tri_in     = TorsionAwareTriangleUpdate(outgoing=False)
        self.pair_trans = PairTransition()
        self.drop_node  = nn.Dropout(cfg.DROPOUT)
        self.drop_pair  = nn.Dropout(cfg.DROPOUT)

    def forward(self, single, pair, torsion, mask=None):
        single = self.row_attn(single, pair, mask)
        single = self.node_trans(single)
        pair   = self.tri_out(pair, torsion)
        pair   = self.tri_in(pair,  torsion)
        pair   = self.pair_trans(pair)
        return single, pair


# ─────────────────────────────────────────────────────────────
# IPA & Structure Module (from v2, enhanced with torsion output)
# ─────────────────────────────────────────────────────────────
class InvariantPointAttention(nn.Module):
    def __init__(self):
        super().__init__()
        D = cfg.D_NODE; H = cfg.N_HEAD; Qp = cfg.N_QUERY_PT; Vp = cfg.N_VALUE_PT
        self.H = H; self.Qp = Qp; self.Vp = Vp
        self.scale_attn  = (D // H) ** -0.5
        self.scale_point = (3 * Qp) ** -0.5
        w_C = math.log(math.exp(1) - 1)
        self.w_C = nn.Parameter(torch.full((H,), w_C))
        self.norm = LayerNorm(D)
        self.q    = Linear(D, H * D // H, bias=False)
        self.k    = Linear(D, H * D // H, bias=False)
        self.v    = Linear(D, H * D // H, bias=False)
        self.q_pt = Linear(D, H * Qp * 3, bias=False)
        self.k_pt = Linear(D, H * Qp * 3, bias=False)
        self.v_pt = Linear(D, H * Vp * 3, bias=False)
        self.b    = Linear(cfg.D_PAIR, H, bias=False)
        out_dim   = H * (D // H + cfg.D_PAIR + Vp * 4 + 1)
        self.out  = Linear(out_dim, D)

    def _apply_frames(self, T, pts):
        R = T[..., :3, :3]; t = T[..., :3, 3]
        return (R.unsqueeze(-2) @ pts.unsqueeze(-1)).squeeze(-1) + t.unsqueeze(-2)

    def _inv_apply_frames(self, T, pts):
        R = T[..., :3, :3]; t = T[..., :3, 3]
        return (R.transpose(-2,-1).unsqueeze(-2) @ (pts - t.unsqueeze(-2)).unsqueeze(-1)).squeeze(-1)

    def forward(self, single, pair, T):
        B, L, D = single.shape
        H = self.H; Dh = D // H
        z = self.norm(single)
        Q = self.q(z).view(B, L, H, Dh)
        K = self.k(z).view(B, L, H, Dh)
        V = self.v(z).view(B, L, H, Dh)
        Q_pt = self.q_pt(z).view(B, L, H, self.Qp, 3)
        K_pt = self.k_pt(z).view(B, L, H, self.Qp, 3)
        V_pt = self.v_pt(z).view(B, L, H, self.Vp, 3)
        Q_g  = self._apply_frames(T.unsqueeze(-3).unsqueeze(-3), Q_pt)
        K_g  = self._apply_frames(T.unsqueeze(-3).unsqueeze(-3), K_pt)
        V_g  = self._apply_frames(T.unsqueeze(-3).unsqueeze(-3), V_pt)
        s_attn = (Q.transpose(1,2) @ K.transpose(1,2).transpose(-2,-1)) * self.scale_attn
        b_attn = self.b(pair).permute(0,3,1,2)
        d2 = ((Q_g.unsqueeze(2) - K_g.unsqueeze(1)) ** 2).sum(-1).sum(-1).permute(0,3,1,2)
        w_pt = F.softplus(self.w_C).view(1, H, 1, 1)
        attn = (s_attn + b_attn - 0.5 * w_pt * d2).softmax(-1)
        o_s  = (attn @ V.transpose(1,2)).transpose(1,2).reshape(B, L, -1)
        o_pt_g = (attn.unsqueeze(-1).unsqueeze(-1) *
                  V_g.unsqueeze(2).permute(0,4,1,2,3)).sum(3).permute(0,2,3,1,4)
        o_pt   = self._inv_apply_frames(T.unsqueeze(-3), o_pt_g)
        o_pt_n = o_pt.norm(dim=-1)
        o_pair = (attn.unsqueeze(-1) * pair.unsqueeze(1)).sum(2).reshape(B, L, -1)
        out    = torch.cat([o_s, o_pair,
                             o_pt.reshape(B, L, -1),
                             o_pt_n.reshape(B, L, -1)], dim=-1)
        return single + self.out(out)


class BackboneUpdate(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = Linear(cfg.D_NODE, 6, init='zeros')

    def forward(self, single, T):
        upd = self.proj(single)
        b, c = upd[..., :3], upd[..., 3:]
        a1 = F.normalize(b, dim=-1)
        a2 = F.normalize(c - (c * a1).sum(-1, keepdim=True) * a1, dim=-1)
        a3 = torch.cross(a1, a2, dim=-1)
        R  = torch.stack([a1, a2, a3], dim=-1)
        t  = upd[..., :3]
        T_new = T.clone()
        T_new[..., :3, :3] = T[..., :3, :3] @ R
        T_new[..., :3,  3] = T[..., :3,  3] + (T[..., :3, :3] @ t.unsqueeze(-1)).squeeze(-1)
        return T_new


class StructureBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = LayerNorm(cfg.D_NODE)
        self.ipa   = InvariantPointAttention()
        self.bb    = BackboneUpdate()
        self.norm2 = LayerNorm(cfg.D_NODE)
        self.ff    = nn.Sequential(
            Linear(cfg.D_NODE, cfg.D_NODE * 4), nn.ReLU(),
            Linear(cfg.D_NODE * 4, cfg.D_NODE, init='final'))
        self.drop  = nn.Dropout(cfg.DROPOUT)

    def forward(self, single, pair, T):
        single = self.norm1(self.drop(self.ipa(single, pair, T)))
        T      = self.bb(single, T)
        single = single + self.drop(self.ff(self.norm2(single)))
        return single, T


# ─────────────────────────────────────────────────────────────
# 5. NOVEL: TORSION ANGLE HEAD
#    Predicts eta, theta, curl as sin/cos pairs per residue.
#    Used both as auxiliary loss and to feed TorsionAwareTriangle.
# ─────────────────────────────────────────────────────────────
class TorsionAngleHead(nn.Module):
    """
    Predicts 3 torsion angles (6 values: sin+cos each).
    Novel: uses both single repr and pair diagonal (self-pair) for prediction.
    """
    def __init__(self):
        super().__init__()
        self.norm = LayerNorm(cfg.D_NODE + cfg.D_PAIR)
        self.mlp  = nn.Sequential(
            Linear(cfg.D_NODE + cfg.D_PAIR, 128), nn.ReLU(),
            Linear(128, 64), nn.ReLU(),
            Linear(64, cfg.N_TORSION_OUT, init='zeros'),
        )

    def forward(self, single: torch.Tensor,
                pair  : torch.Tensor) -> torch.Tensor:
        # pair diagonal: (B, L, D_PAIR)
        B, L = single.shape[:2]
        diag_idx = torch.arange(L, device=pair.device)
        pair_diag = pair[:, diag_idx, diag_idx]               # (B, L, D_PAIR)
        combined  = self.norm(torch.cat([single, pair_diag], dim=-1))
        raw       = self.mlp(combined)                         # (B, L, 6)
        # Normalise sin/cos pairs so they lie on the unit circle
        raw = raw.view(B, L, 3, 2)
        raw = F.normalize(raw, dim=-1)
        return raw.view(B, L, 6)                               # (B, L, 6)


# ─────────────────────────────────────────────────────────────
# 6. NOVEL: pLDDT HEAD
#    Per-residue confidence estimate (50-bin softmax like AF2).
#    Used as: (a) supervision signal, (b) coordinate masking during recycling.
# ─────────────────────────────────────────────────────────────
class pLDDTHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = LayerNorm(cfg.D_NODE)
        self.mlp  = nn.Sequential(
            Linear(cfg.D_NODE, 128), nn.ReLU(),
            Linear(128, cfg.N_PLDDT_BINS),
        )
        # bin centers (0-100)
        centers = torch.linspace(1, 99, cfg.N_PLDDT_BINS)
        self.register_buffer('centers', centers)

    def forward(self, single: torch.Tensor):
        logits   = self.mlp(self.norm(single))                  # (B, L, 50)
        probs    = logits.softmax(dim=-1)
        plddt    = (probs * self.centers.view(1, 1, -1)).sum(-1) # (B, L)  0-100
        return plddt, logits


# ─────────────────────────────────────────────────────────────
# INPUT EMBEDDINGS (enhanced from v2)
# ─────────────────────────────────────────────────────────────
class NodeEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_embed    = nn.Embedding(cfg.VOCAB_SIZE, 64, padding_idx=4)
        self.f1_proj      = Linear(cfg.F1_DIM, 32)
        self.dihed_proj   = Linear(cfg.N_DIHED, 32)
        self.ss_proj      = Linear(1, 16)
        self.laplacian_proj = Linear(cfg.K_LAPLACIAN, 32)   # NEW: from templates
        self.pe           = SinusoidalPE(cfg.D_NODE, cfg.MAX_LEN)
        self.out_proj     = nn.Sequential(
            Linear(64 + 32 + 32 + 16 + 32, cfg.D_NODE), nn.ReLU(), LayerNorm(cfg.D_NODE))

    def forward(self, seq_ids, f1, dihed, ss_pair, tmpl_laplacian_mean):
        """
        tmpl_laplacian_mean: (B, L, K_LAPLACIAN) — mean Laplacian eigvecs over templates
        """
        s = self.seq_embed(seq_ids)
        f = self.f1_proj(f1)
        d = self.dihed_proj(dihed)
        p = self.ss_proj(ss_pair.unsqueeze(-1))
        e = self.laplacian_proj(tmpl_laplacian_mean)
        x = torch.cat([s, f, d, p, e], dim=-1)
        x = self.out_proj(x)
        return self.pe(x)


class PairEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        always_dim = cfg.N_REL_POS + 2 + 1 + cfg.N_PAIR_TYPE
        geo_dim    = cfg.N_RBF + cfg.N_DIST_BINS + cfg.N_ORIENT
        in_dim     = always_dim + geo_dim
        self.proj  = nn.Sequential(
            Linear(in_dim, cfg.D_PAIR * 2), nn.ReLU(),
            Linear(cfg.D_PAIR * 2, cfg.D_PAIR), LayerNorm(cfg.D_PAIR))

    def forward(self, rbf, dist_bins, orient, rel_pos,
                MIp, FNp, contact_ss, pair_type):
        cov = torch.stack([MIp, FNp], dim=-1)
        if dist_bins.dim() == 3:
            dist_bins = F.one_hot(
                dist_bins.long().clamp(0, cfg.N_DIST_BINS - 1),
                num_classes=cfg.N_DIST_BINS).float()
        x = torch.cat([rbf, dist_bins, orient, rel_pos, cov,
                        contact_ss.unsqueeze(-1), pair_type], dim=-1)
        return self.proj(x)


# ─────────────────────────────────────────────────────────────
# AUXILIARY HEADS
# ─────────────────────────────────────────────────────────────
class DistogramHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = Linear(cfg.D_PAIR, cfg.N_DIST_BINS)
    def forward(self, pair):
        return self.proj(pair)

class ContactHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(Linear(cfg.D_PAIR, 64), nn.ReLU(),
                                   Linear(64, 1), nn.Sigmoid())
    def forward(self, pair):
        return self.proj(pair).squeeze(-1)

class CoordinateHead(nn.Module):
    def forward(self, T):
        return T[..., :3, 3]


# ─────────────────────────────────────────────────────────────
# FULL MODEL
# ─────────────────────────────────────────────────────────────
class RNAFoldTemplate(nn.Module):
    """
    Template-guided RNA 3D folding model.

    New data flow:
      1. Template features are pre-fused into initial pair/single repr
         via TemplatePairStack and TemplateSingleStack.
      2. Each EvoformerBlock uses TorsionAwareTriangleUpdate with
         running torsion angle predictions.
      3. SparseGraphAttention applied every other block using
         current coordinate estimate.
      4. pLDDT head produces per-residue confidence; low-confidence
         residues are down-weighted in the coordinate loss (focal-style).
      5. TorsionAngleHead produces auxiliary torsion supervision.
    """
    def __init__(self):
        super().__init__()
        self.node_embed = NodeEmbedding()
        self.pair_embed = PairEmbedding()

        # Template modules
        self.template_pair_stack   = TemplatePairStack()
        self.template_single_stack = TemplateSingleStack()

        # Evoformer with torsion-aware triangles
        self.evoformer = nn.ModuleList([EvoformerBlock() for _ in range(cfg.N_EVOFORMER)])

        # Sparse graph attention (applied every 2 blocks)
        self.graph_attn = SparseGraphAttention()

        # Structure module
        self.structure = nn.ModuleList([StructureBlock() for _ in range(cfg.N_STRUCTURE)])

        # Heads
        self.dist_head    = DistogramHead()
        self.contact_head = ContactHead()
        self.coord_head   = CoordinateHead()
        self.torsion_head = TorsionAngleHead()
        self.plddt_head   = pLDDTHead()

        # Recycling projections
        self.recycle_single = Linear(cfg.D_NODE, cfg.D_NODE)
        self.recycle_pair   = Linear(cfg.D_PAIR,  cfg.D_PAIR)
        # Recycling: pLDDT-gated coordinate bias
        self.recycle_coord_proj = Linear(3, cfg.D_PAIR)

    def init_frames(self, B, L, device):
        T = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0)
        T = T.expand(B, L, -1, -1).clone()
        T[:, :, 2, 3] = torch.arange(L, device=device).float().unsqueeze(0) * 6.0
        return T

    def forward(self, batch, device=None):
        if device is None:
            device = next(self.parameters()).device

        def to(x):
            return x.to(device) if isinstance(x, torch.Tensor) else x

        # Standard inputs
        seq_ids    = to(batch['seq_ids'])
        seq_mask   = to(batch['seq_mask'])
        f1         = to(batch['f1'])
        dihed      = to(batch['dihed'])
        ss_pair    = to(batch['ss_pair'])
        rbf        = to(batch['dist_rbf'])
        dist_bins  = to(batch['dist_bins'])
        orient     = to(batch['orient'])
        rel_pos    = to(batch['rel_pos'])
        MIp        = to(batch['MIp'])
        FNp        = to(batch['FNp'])
        contact_ss = to(batch['contact_ss'])
        pair_type  = to(batch['pair_type'])

        # Template inputs (new)
        tmpl_dgram     = to(batch['tmpl_dgram'])      # (B, T, L, L, 38)
        tmpl_frames    = to(batch['tmpl_frames'])     # (B, T, L, 4, 4)
        tmpl_torsion   = to(batch['tmpl_torsion'])    # (B, T, L, 6)
        tmpl_laplacian = to(batch['tmpl_laplacian'])  # (B, T, L, 8)
        tmpl_seq_match = to(batch['tmpl_seq_match'])  # (B, T, L)
        tmpl_valid     = to(batch['tmpl_valid'])      # (B, T, L)
        tmpl_weights   = to(batch['tmpl_weights'])    # (B, T)

        B, L = seq_ids.shape

        # Mean Laplacian eigvecs for node embedding
        w    = tmpl_weights.softmax(1).view(B, -1, 1, 1)
        tmpl_lap_mean = (tmpl_laplacian * w).sum(1)            # (B, L, 8)

        # Initial embeddings
        single = self.node_embed(seq_ids, f1, dihed, ss_pair, tmpl_lap_mean)
        pair   = self.pair_embed(rbf, dist_bins, orient, rel_pos,
                                  MIp, FNp, contact_ss, pair_type)

        # Inject template information into pair and single representations
        pair   = self.template_pair_stack(pair, tmpl_dgram, tmpl_laplacian,
                                           tmpl_weights, tmpl_valid)
        single = self.template_single_stack(single, tmpl_frames, tmpl_torsion,
                                             tmpl_weights, tmpl_seq_match)

        # Initial torsion prediction
        torsion_pred = self.torsion_head(single, pair)

        # Recycling
        prev_single = torch.zeros_like(single)
        prev_pair   = torch.zeros_like(pair)
        T           = self.init_frames(B, L, device)
        all_coords  = []
        all_plddt   = []

        for recycle_idx in range(cfg.N_RECYCLE):
            single = single + self.recycle_single(prev_single)
            pair   = pair   + self.recycle_pair(prev_pair)

            # pLDDT-gated coordinate bias: weight recycled coords by confidence
            if recycle_idx > 0 and len(all_plddt) > 0:
                prev_conf = all_plddt[-1].detach() / 100.0          # (B, L)
                coord_prev = all_coords[-1].detach()                  # (B, L, 3)
                coord_bias = self.recycle_coord_proj(coord_prev)      # (B, L, D_PAIR)
                # Add as diagonal pair bias (self-interaction)
                idx = torch.arange(L, device=device)
                pair[:, idx, idx] = pair[:, idx, idx] + coord_bias * prev_conf.unsqueeze(-1)

            # Evoformer with sparse graph attention interspersed
            for blk_idx, block in enumerate(self.evoformer):
                if self.training:
                    single, pair = torch.utils.checkpoint.checkpoint(
                        lambda s, p: block(s, p, torsion_pred, seq_mask),
                        single, pair, use_reentrant=False)
                else:
                    single, pair = block(single, pair, torsion_pred, seq_mask)

                # Sparse graph attention every 2 blocks using current coordinate guess
                if blk_idx % 2 == 1:
                    coords_now = T[..., :3, 3]
                    single     = self.graph_attn(single, pair, coords_now, contact_ss)

                # Update torsion prediction mid-trunk
                if blk_idx == cfg.N_EVOFORMER // 2:
                    torsion_pred = self.torsion_head(single, pair)

            # Structure module
            s = single
            for block in self.structure:
                if self.training:
                    s, T = torch.utils.checkpoint.checkpoint(
                        block, s, pair, T, use_reentrant=False)
                else:
                    s, T = block(s, pair, T)

            prev_single = single.detach()
            prev_pair   = pair.detach()
            single      = s

            coords = self.coord_head(T)
            plddt, plddt_logits = self.plddt_head(single)
            all_coords.append(coords)
            all_plddt.append(plddt)

        # Final torsion
        torsion_final = self.torsion_head(single, pair)
        distogram     = self.dist_head(pair)
        contact_pred  = self.contact_head(pair)
        _, plddt_logits_final = self.plddt_head(single)

        return {
            'coords'         : all_coords[-1],
            'all_coords'     : all_coords,
            'distogram'      : distogram,
            'contact'        : contact_pred,
            'torsion'        : torsion_final,        # (B, L, 6) — sin/cos
            'plddt'          : all_plddt[-1],         # (B, L) — 0-100
            'plddt_logits'   : plddt_logits_final,    # (B, L, 50)
            'pair'           : pair,
            'single'         : single,
        }


def build_model_dual_gpu():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = RNAFoldTemplate()
    n_gpus = torch.cuda.device_count()
    if n_gpus >= 2:
        print(f"  🔥  Using {n_gpus} GPUs via DataParallel")
        model = nn.DataParallel(model, device_ids=list(range(n_gpus)))
    elif n_gpus == 1:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    model = model.to(device)
    return model, device


# ─────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = RNAFoldTemplate().to(device)
    n_p    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device}  |  Parameters: {n_p:,}")

    B, L, T = 2, 64, 4
    fake = {
        'seq_ids'   : torch.randint(0, 4, (B, L)).to(device),
        'seq_mask'  : torch.ones(B, L, dtype=torch.bool).to(device),
        'f1'        : torch.rand(B, L, 5).to(device),
        'dihed'     : torch.zeros(B, L, 4).to(device),
        'ss_pair'   : torch.rand(B, L).to(device),
        'dist_rbf'  : torch.zeros(B, L, L, 16).to(device),
        'dist_bins' : torch.zeros(B, L, L, 38).to(device),
        'orient'    : torch.zeros(B, L, L, 4).to(device),
        'rel_pos'   : torch.rand(B, L, L, 65).to(device),
        'MIp'       : torch.rand(B, L, L).to(device),
        'FNp'       : torch.rand(B, L, L).to(device),
        'contact_ss': torch.rand(B, L, L).to(device),
        'pair_type' : torch.rand(B, L, L, 3).to(device),
        # Template inputs
        'tmpl_dgram'    : torch.rand(B, T, L, L, 38).to(device),
        'tmpl_frames'   : torch.eye(4).view(1,1,1,4,4).expand(B,T,L,-1,-1).clone().to(device),
        'tmpl_torsion'  : torch.rand(B, T, L, 6).to(device),
        'tmpl_laplacian': torch.rand(B, T, L, 8).to(device),
        'tmpl_seq_match': torch.rand(B, T, L).to(device),
        'tmpl_valid'    : torch.ones(B, T, L).to(device),
        'tmpl_weights'  : torch.ones(B, T).to(device) / T,
    }

    with torch.no_grad():
        out = model(fake, device=device)

    print(f"coords      : {out['coords'].shape}")
    print(f"distogram   : {out['distogram'].shape}")
    print(f"torsion     : {out['torsion'].shape}")
    print(f"plddt       : {out['plddt'].shape}  range=[{out['plddt'].min():.1f}, {out['plddt'].max():.1f}]")
    print("Forward pass OK ✅  (template-guided inference)")
