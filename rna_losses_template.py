"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  NOVEL LOSS FUNCTIONS — TEMPLATE-GUIDED RNA FOLDING                          ║
║                                                                               ║
║  New losses (vs v2/v3):                                                       ║
║   1. pLDDT distillation loss  — align predicted confidence to actual LDDT    ║
║   2. Torsion angle loss       — MAE on sin/cos eta/theta/curl                 ║
║   3. Template-aware FAPE      — FAPE computed in template frame for guidance ║
║   4. Focal coordinate loss    — down-weight low-quality residues (pLDDT <50) ║
║   5. LDDT-Cα auxiliary loss   — direct LDDT-like metric as differentiable    ║
║                                  training signal                              ║
║   6. Distogram cross-entropy  — soft triangle-bin cross-entropy               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import math, torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


# ═════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════
def kabsch_rmsd(P: torch.Tensor, Q: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Differentiable Kabsch RMSD. P, Q: (B, L, 3). mask: (B, L) bool.
    Returns (B,) RMSD.
    """
    B, L, _ = P.shape
    if mask is None:
        mask = torch.ones(B, L, dtype=torch.bool, device=P.device)

    loss_per = []
    for b in range(B):
        m  = mask[b]
        Pb = P[b][m] - P[b][m].mean(0)
        Qb = Q[b][m] - Q[b][m].mean(0)
        H  = Pb.T @ Qb
        U, S, Vt = torch.linalg.svd(H)
        D  = torch.diag(torch.tensor([1., 1.,
              torch.det(Vt.T @ U.T).sign().clamp(min=-1, max=1)],
              device=P.device))
        R  = Vt.T @ D @ U.T
        Pb_rot = Pb @ R.T
        rmsd   = (((Pb_rot - Qb) ** 2).sum(-1).mean() + 1e-8).sqrt()
        loss_per.append(rmsd)
    return torch.stack(loss_per)


def tm_score_batch(pred: torch.Tensor, true: torch.Tensor,
                   mask: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
    """Differentiable approximate TM-score. Returns (B,) scores."""
    B = pred.shape[0]
    scores = []
    for b in range(B):
        L  = int(seq_len[b])
        d0 = max(1.24 * (L - 15) ** (1/3) - 1.8, 0.5) if L > 21 else 0.5
        m  = mask[b, :L].bool()
        if m.sum() < 3:
            scores.append(pred.new_zeros(1).squeeze())
            continue
        p = pred[b, :L][m] - pred[b, :L][m].mean(0)
        t = true[b, :L][m] - true[b, :L][m].mean(0)
        tm = (1 / (1 + ((p - t) ** 2).sum(-1) / (d0 ** 2 + 1e-8))).mean()
        scores.append(tm)
    return torch.stack(scores)


# ═════════════════════════════════════════════════════════════
# 1. FAPE LOSS (Frame-Aligned Point Error)
# ═════════════════════════════════════════════════════════════
def fape_loss(pred_coords : torch.Tensor,   # (B, L, 3)
              true_coords : torch.Tensor,   # (B, L, 3)
              pred_frames : torch.Tensor,   # (B, L, 4, 4)
              true_frames : torch.Tensor,   # (B, L, 4, 4)
              mask        : torch.Tensor,   # (B, L)
              clamp_dist  : float = 10.0,
              ) -> torch.Tensor:
    """
    FAPE: compare predicted coords expressed in each predicted frame
    to true coords expressed in corresponding true frames.
    """
    B, L, _ = pred_coords.shape

    def apply_inv_frame(frames, coords):
        R = frames[..., :3, :3]      # (B, L, 3, 3)
        t = frames[..., :3,  3]      # (B, L, 3)
        # coords: (B, L', 3) → express in frame of each residue i
        # result: (B, L, L', 3)
        c_shift = coords.unsqueeze(1) - t.unsqueeze(2)       # (B, L, L', 3)
        return (R.unsqueeze(2).transpose(-2,-1) @
                c_shift.unsqueeze(-1)).squeeze(-1)            # (B, L, L', 3)

    pred_local = apply_inv_frame(pred_frames, pred_coords)    # (B, L, L, 3)
    true_local = apply_inv_frame(true_frames, true_coords)

    diff  = (pred_local - true_local).norm(dim=-1)            # (B, L, L)
    diff  = torch.clamp(diff, max=clamp_dist)

    m2d   = mask.unsqueeze(1) * mask.unsqueeze(2)             # (B, L, L)
    n     = m2d.sum() + 1e-8
    return (diff * m2d).sum() / n


# ═════════════════════════════════════════════════════════════
# 2. NOVEL: pLDDT DISTILLATION LOSS
#    Computes actual per-residue lDDT-Cα for each predicted structure
#    and regresses the pLDDT head's logits towards it.
# ═════════════════════════════════════════════════════════════
def compute_lddt_ca(pred  : torch.Tensor,   # (B, L, 3)
                    true  : torch.Tensor,   # (B, L, 3)
                    mask  : torch.Tensor,   # (B, L)
                    cutoff: float = 15.0,
                    ) -> torch.Tensor:
    """
    Compute per-residue lDDT-Cα (0-1) in a differentiable-friendly way.
    For each residue i, measures agreement on all pairwise C1' distances within cutoff.
    Returns (B, L) score in [0, 1].
    """
    B, L = pred.shape[:2]
    pred_d = torch.cdist(pred, pred)          # (B, L, L)
    true_d = torch.cdist(true, true)

    # Select pairs within cutoff in true structure
    in_cut = (true_d < cutoff) & (true_d > 0)  # (B, L, L)
    m2d    = mask.unsqueeze(1) * mask.unsqueeze(2)
    in_cut = in_cut & m2d.bool()

    diff   = (pred_d - true_d).abs()           # (B, L, L)
    thresholds = [0.5, 1.0, 2.0, 4.0]
    per_res = torch.zeros(B, L, device=pred.device)
    for thr in thresholds:
        preserved = (diff < thr).float() * in_cut.float()
        denom     = in_cut.float().sum(-1).clamp(min=1)
        per_res   = per_res + preserved.sum(-1) / denom
    per_res = per_res / len(thresholds)
    return per_res * mask.float()              # (B, L)


def plddt_distillation_loss(plddt_logits: torch.Tensor,   # (B, L, 50)
                             pred_coords : torch.Tensor,   # (B, L, 3)
                             true_coords : torch.Tensor,   # (B, L, 3)
                             mask        : torch.Tensor,   # (B, L)
                             ) -> torch.Tensor:
    """
    Compute actual lDDT-Cα, discretise to 50 bins, cross-entropy against logits.
    Novel: teaches confidence head to be calibrated, not just self-consistent.
    """
    with torch.no_grad():
        lddt = compute_lddt_ca(pred_coords.detach(), true_coords, mask)  # (B, L)
        # Discretise 0-1 → 50 bins (like AF2 uses 0-100 / 2 bins)
        bin_idx = (lddt * 49).long().clamp(0, 49)

    B, L, n_bins = plddt_logits.shape
    loss = F.cross_entropy(
        plddt_logits.view(B * L, n_bins),
        bin_idx.view(B * L),
        reduction='none',
    ).view(B, L)
    return (loss * mask.float()).sum() / (mask.float().sum() + 1e-8)


# ═════════════════════════════════════════════════════════════
# 3. NOVEL: TORSION ANGLE LOSS
#    MAE on sin/cos pairs (unit-circle loss).
# ═════════════════════════════════════════════════════════════
def torsion_angle_loss(pred_torsion: torch.Tensor,   # (B, L, 6)
                       true_torsion: torch.Tensor,   # (B, L, 6)
                       mask        : torch.Tensor,   # (B, L)
                       ) -> torch.Tensor:
    """
    Unit-circle MAE: for each sin/cos pair, measures angular distance.
    Novel: normalises pred to unit circle before comparison, stable gradients.
    """
    # Reshape as (B, L, 3, 2) sin/cos pairs
    p = pred_torsion.view(*pred_torsion.shape[:2], 3, 2)
    t = true_torsion.view(*true_torsion.shape[:2], 3, 2)
    p = F.normalize(p, dim=-1)   # project to unit circle
    t = F.normalize(t, dim=-1)

    # Angular distance: 1 − cos(θ_pred − θ_true) = 1 − (p·t)
    cos_sim = (p * t).sum(-1)          # (B, L, 3)
    loss    = (1.0 - cos_sim).mean(-1) # (B, L)  in [0, 2]
    loss    = loss / 2.0               # normalise to [0, 1]
    return (loss * mask.float()).sum() / (mask.float().sum() + 1e-8)


# ═════════════════════════════════════════════════════════════
# 4. NOVEL: TEMPLATE-AWARE FAPE
#    FAPE computed in the frame of the best-matching template.
#    Encourages the model to produce predictions consistent with
#    the template geometry for high-similarity regions.
# ═════════════════════════════════════════════════════════════
def template_fape_loss(pred_coords   : torch.Tensor,   # (B, L, 3)
                       tmpl_frames   : torch.Tensor,   # (B, T, L, 4, 4)
                       tmpl_weights  : torch.Tensor,   # (B, T)
                       tmpl_valid    : torch.Tensor,   # (B, T, L)
                       ) -> torch.Tensor:
    """
    For each template, compute the coordinate prediction expressed in
    template frames. Weighted by template confidence, this encourages
    alignment with high-confidence template regions.
    Novel: uses learnable soft-weighting instead of hard template selection.
    """
    B, T, L, _, _ = tmpl_frames.shape
    w = tmpl_weights.softmax(dim=1)               # (B, T)

    loss_total = pred_coords.new_zeros(1)
    for t_idx in range(T):
        frames = tmpl_frames[:, t_idx]             # (B, L, 4, 4)
        valid  = tmpl_valid[:, t_idx]              # (B, L)
        R = frames[..., :3, :3]                    # (B, L, 3, 3)
        tr = frames[..., :3, 3]                    # (B, L, 3)

        # Express pred_coords in each template frame
        c_shift = pred_coords.unsqueeze(1) - tr.unsqueeze(2)    # (B, L, L, 3)
        local_c = (R.unsqueeze(2).transpose(-2,-1) @
                   c_shift.unsqueeze(-1)).squeeze(-1)            # (B, L, L, 3)

        # Symmetric: norm of residuals in frame (should be small if coords
        # match template geometry in local frame)
        residuals = local_c.norm(dim=-1)                         # (B, L, L)
        residuals = torch.clamp(residuals, max=10.0)

        m2d = valid.unsqueeze(1) * valid.unsqueeze(2)            # (B, L, L)
        n   = m2d.sum() + 1e-8
        fape_t = (residuals * m2d).sum() / n

        loss_total = loss_total + w[:, t_idx].mean() * fape_t

    return loss_total / T


# ═════════════════════════════════════════════════════════════
# 5. NOVEL: FOCAL COORDINATE LOSS
#    pLDDT-weighted MSE: down-weight low-confidence predictions.
#    Focal-like: high-confidence errors get amplified, preventing
#    the model from ignoring hard residues.
# ═════════════════════════════════════════════════════════════
def focal_coordinate_loss(pred_coords: torch.Tensor,   # (B, L, 3)
                           true_coords: torch.Tensor,   # (B, L, 3)
                           plddt      : torch.Tensor,   # (B, L) in [0, 100]
                           mask       : torch.Tensor,   # (B, L)
                           gamma      : float = 2.0,
                           ) -> torch.Tensor:
    """
    Focal-weighted MSE:
      loss_i = (plddt_i/100)^gamma × MSE(pred_i, true_i)

    High confidence + large error → amplified penalty (focal effect).
    Low confidence → reduced penalty (prevents forcing bad predictions).
    Novel: unlike standard RMSD, explicitly couples confidence to accuracy.
    """
    conf   = (plddt / 100.0).detach()                 # (B, L) in [0, 1]
    weight = conf.pow(gamma)                           # high-conf → high weight

    per_res_mse = ((pred_coords - true_coords) ** 2).sum(-1)   # (B, L)
    loss        = weight * per_res_mse
    return (loss * mask.float()).sum() / (mask.float().sum() + 1e-8)


# ═════════════════════════════════════════════════════════════
# 6. NOVEL: LDDT-Cα AUXILIARY LOSS
#    Direct differentiable approximation of LDDT-Cα using a sigmoid
#    approximation of the indicator function.
# ═════════════════════════════════════════════════════════════
def lddt_loss(pred_coords: torch.Tensor,   # (B, L, 3)
              true_coords: torch.Tensor,   # (B, L, 3)
              mask       : torch.Tensor,   # (B, L)
              cutoff     : float = 15.0,
              sigma      : float = 0.5,
              ) -> torch.Tensor:
    """
    Soft LDDT-Cα: sigmoid approximation to the threshold indicator.
    Maximise sum of sigmoid((thr − |d_pred − d_true|) / sigma)
    Novel: directly optimises LDDT-like metric rather than a proxy.
    """
    B, L = pred_coords.shape[:2]
    pred_d = torch.cdist(pred_coords, pred_coords)
    true_d = torch.cdist(true_coords, true_coords)

    in_cut = (true_d < cutoff) & (true_d > 0)
    m2d    = mask.unsqueeze(1) * mask.unsqueeze(2)
    in_cut = in_cut & m2d.bool()

    diff = (pred_d - true_d).abs()
    n_pairs = in_cut.float().sum() + 1e-8

    # Soft indicator: big when diff is small
    soft_lddt = 0.0
    for thr in [0.5, 1.0, 2.0, 4.0]:
        soft_lddt = soft_lddt + torch.sigmoid((thr - diff) / sigma)
    soft_lddt = soft_lddt / 4.0 * in_cut.float()

    # Loss = 1 - mean lDDT (so we minimise it)
    return 1.0 - soft_lddt.sum() / n_pairs


# ═════════════════════════════════════════════════════════════
# 7. DISTOGRAM LOSS (soft triangle bins)
# ═════════════════════════════════════════════════════════════
def distogram_loss(pred_logits : torch.Tensor,   # (B, L, L, n_bins)
                   true_dist   : torch.Tensor,   # (B, L, L) — raw distances
                   mask        : torch.Tensor,   # (B, L)
                   n_bins      : int = 38,
                   d_min       : float = 2.0,
                   d_max       : float = 22.0,
                   ) -> torch.Tensor:
    """Cross-entropy with triangular soft target bins."""
    B, L, _, nb = pred_logits.shape
    edges   = torch.linspace(d_min, d_max, nb + 1, device=pred_logits.device)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width   = (d_max - d_min) / nb

    d_exp = true_dist.unsqueeze(-1)                        # (B, L, L, 1)
    tri   = torch.clamp(1 - (d_exp - centers).abs() / width, min=0)
    tri   = tri / (tri.sum(-1, keepdim=True) + 1e-8)      # soft target

    m2d   = (mask.unsqueeze(1) * mask.unsqueeze(2)).bool()
    log_p = F.log_softmax(pred_logits, dim=-1)
    xent  = -(tri * log_p).sum(-1)                        # (B, L, L)
    return (xent * m2d.float()).sum() / (m2d.float().sum() + 1e-8)


# ═════════════════════════════════════════════════════════════
# COMBINED LOSS
# ═════════════════════════════════════════════════════════════
def compute_total_loss(
    outputs    : Dict[str, torch.Tensor],
    batch      : Dict[str, torch.Tensor],
    cfg_weights: object,
    device     : torch.device,
) -> tuple:
    """
    Compute the combined training loss.
    Returns (total_loss, loss_dict).
    """
    pred_all   = outputs['all_coords']
    true_coords = batch.get('true_coords')
    if true_coords is None:
        return outputs['coords'].new_zeros(1), {}
    true_coords = true_coords.to(device)

    mask     = batch['seq_mask'].to(device)
    seq_len  = batch['seq_len'].to(device)
    plddt    = outputs['plddt']

    # Build frame tensors from true coords for FAPE
    # (simplified: use identity frames as true frames for standard FAPE)
    B, L = mask.shape
    true_frames = torch.eye(4, device=device).view(1, 1, 4, 4).expand(B, L, 4, 4).clone()
    true_frames[:, :, :3, 3] = true_coords

    pred_frames_last = torch.eye(4, device=device).view(1, 1, 4, 4).expand(B, L, 4, 4).clone()
    pred_frames_last[:, :, :3, 3] = pred_all[-1]

    losses = {}

    # ── Focal coordinate loss (novel) ─────────────────────
    losses['focal'] = focal_coordinate_loss(
        pred_all[-1], true_coords, plddt, mask, gamma=2.0)

    # ── TM-score loss ──────────────────────────────────────
    tm_scores = tm_score_batch(pred_all[-1], true_coords, mask, seq_len)
    losses['tm'] = -tm_scores.mean()

    # ── FAPE ──────────────────────────────────────────────
    losses['fape'] = fape_loss(
        pred_all[-1], true_coords, pred_frames_last, true_frames, mask)

    # ── LDDT-Cα auxiliary loss (novel) ────────────────────
    losses['lddt'] = lddt_loss(pred_all[-1], true_coords, mask)

    # ── pLDDT distillation loss (novel) ───────────────────
    losses['plddt'] = plddt_distillation_loss(
        outputs['plddt_logits'], pred_all[-1].detach(), true_coords, mask)

    # ── Torsion angle loss (novel) ────────────────────────
    true_torsion = batch.get('true_torsion')
    if true_torsion is not None:
        true_torsion = true_torsion.to(device)
        losses['torsion'] = torsion_angle_loss(outputs['torsion'], true_torsion, mask)
    else:
        losses['torsion'] = outputs['coords'].new_zeros(1)

    # ── Template-aware FAPE (novel) ───────────────────────
    tmpl_frames  = batch.get('tmpl_frames')
    tmpl_weights = batch.get('tmpl_weights')
    tmpl_valid   = batch.get('tmpl_valid')
    if all(x is not None for x in [tmpl_frames, tmpl_weights, tmpl_valid]):
        losses['tmpl_fape'] = template_fape_loss(
            pred_all[-1],
            tmpl_frames.to(device),
            tmpl_weights.to(device),
            tmpl_valid[:, :, :L].to(device),
        )
    else:
        losses['tmpl_fape'] = outputs['coords'].new_zeros(1)

    # ── Distogram ─────────────────────────────────────────
    true_dist_mat = torch.cdist(true_coords, true_coords)
    losses['distogram'] = distogram_loss(outputs['distogram'], true_dist_mat, mask)

    # ── Recycling intermediate loss ────────────────────────
    recycle_loss = outputs['coords'].new_zeros(1)
    for coords_r in pred_all[:-1]:
        recycle_loss = recycle_loss + F.mse_loss(
            coords_r * mask.unsqueeze(-1).float(),
            true_coords * mask.unsqueeze(-1).float())
    recycle_loss = recycle_loss / max(len(pred_all) - 1, 1)
    losses['recycle'] = recycle_loss

    # ── Weighted total ────────────────────────────────────
    w = cfg_weights
    total = (
          w.W_COORD     * losses['focal']
        + w.W_TM        * losses['tm']
        + w.W_FAPE      * losses['fape']
        + w.W_LDDT      * losses['lddt']
        + w.W_PLDDT     * losses['plddt']
        + w.W_TORSION   * losses['torsion']
        + w.W_TMPL_FAPE * losses['tmpl_fape']
        + w.W_DIST      * losses['distogram']
        + w.W_RECYCLE   * losses['recycle']
    )
    return total, losses
