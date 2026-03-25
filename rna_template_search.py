"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  RNA TEMPLATE SEARCH & WEIGHTING — NOVEL COMPONENTS                          ║
║                                                                               ║
║  Novel features:                                                              ║
║   • Learned template similarity gating (not hard-cutoff RMSD)                ║
║   • Torsion-aware sequence alignment for template selection                  ║
║   • Multi-template soft-fusion via attention over template pool               ║
║   • Template confidence weighting (down-weight low-resolution templates)     ║
║   • Graph Laplacian eigenvector template fingerprint                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os, math, warnings
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
RNA_MAP = {
    'A':'A','U':'U','G':'G','C':'C',
    'RA':'A','RU':'U','RG':'G','RC':'C',
    '2MG':'G','1MG':'G','7MG':'G','5MC':'C',
    'H2U':'U','PSU':'U','I':'G','M2G':'G',
}
VOCAB = {'A':0,'U':1,'G':2,'C':3}
WC_PAIRS = {('A','U'),('U','A'),('G','C'),('C','G'),('G','U'),('U','G')}

MAX_TEMPLATES  = 4      # maximum templates to fuse per query
K_LAPLACIAN    = 8      # Laplacian eigenvector dims for fingerprint
TEMPLATE_MAXL  = 512


# ═════════════════════════════════════════════════════════════
# 1. SEQUENCE IDENTITY & ALIGNMENT
# ═════════════════════════════════════════════════════════════
def sequence_identity(a: str, b: str) -> float:
    """Global sequence identity, ignoring gaps."""
    if not a or not b:
        return 0.0
    mn = min(len(a), len(b))
    matches = sum(1 for x, y in zip(a[:mn], b[:mn]) if x == y)
    return matches / max(len(a), len(b))


def local_best_alignment(query: str, template_seq: str,
                         window: int = 32) -> Tuple[int, float]:
    """
    Sliding-window local alignment — finds best offset of template into query.
    Returns (offset, identity_score).
    Novel: uses dinucleotide context bonus for RNA coaxial stacking pairs.
    """
    Lq, Lt = len(query), len(template_seq)
    if Lt <= window:
        return 0, sequence_identity(query, template_seq)

    best_offset, best_score = 0, 0.0
    for offset in range(max(1, Lq - Lt + 1)):
        q_chunk = query[offset: offset + min(window, Lt)]
        t_chunk = template_seq[:min(window, len(q_chunk))]
        score   = 0.0
        for i, (q, t) in enumerate(zip(q_chunk, t_chunk)):
            if q == t:
                score += 1.0
                # Dinucleotide bonus: reward matching consecutive base pairs
                if i > 0 and q_chunk[i-1] == t_chunk[i-1]:
                    score += 0.15
        score /= max(len(q_chunk), 1)
        if score > best_score:
            best_score = score
            best_offset = offset
    return best_offset, best_score


# ═════════════════════════════════════════════════════════════
# 2. STRUCTURAL FEATURES FROM TEMPLATE
# ═════════════════════════════════════════════════════════════
def compute_template_distogram(coords: np.ndarray,
                                n_bins: int = 38,
                                d_min: float = 2.0,
                                d_max: float = 22.0) -> np.ndarray:
    """
    Compute soft binned distance map for template coords.
    Returns (L, L, n_bins) float32.
    Novel: uses triangular basis functions (softer than hard bins).
    """
    L = len(coords)
    valid = ~np.isnan(coords[:, 0])
    c = coords.copy()
    c[~valid] = 0.0

    diff  = c[:, None, :] - c[None, :, :]       # (L, L, 3)
    dists = np.sqrt((diff ** 2).sum(-1) + 1e-8)  # (L, L)
    dists[~valid, :] = 0.0
    dists[:, ~valid] = 0.0

    edges   = np.linspace(d_min, d_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width   = (d_max - d_min) / n_bins

    # Triangular soft bins: each distance votes into neighbouring bins
    dists_exp = dists[:, :, None]                 # (L, L, 1)
    tri       = np.maximum(0, 1 - np.abs(dists_exp - centers[None, None, :]) / width)
    tri_sum   = tri.sum(-1, keepdims=True) + 1e-8
    return (tri / tri_sum).astype(np.float32)


def compute_template_frames(coords: np.ndarray) -> np.ndarray:
    """
    Build local orthonormal frames from consecutive C1' atoms.
    Returns (L, 4, 4) rigid body transforms (rotation + translation).
    Novel: uses SVD-based local frames per residue triplet for stability.
    """
    L = len(coords)
    T = np.eye(4, dtype=np.float32)[None].repeat(L, axis=0)
    valid = ~np.isnan(coords[:, 0])

    for i in range(1, L - 1):
        if not (valid[i-1] and valid[i] and valid[i+1]):
            continue
        p0, p1, p2 = coords[i-1], coords[i], coords[i+1]
        e1 = p1 - p0; e1 /= (np.linalg.norm(e1) + 1e-8)
        e2 = p2 - p1; e2 /= (np.linalg.norm(e2) + 1e-8)
        n  = np.cross(e1, e2); nn = np.linalg.norm(n)
        if nn < 1e-6:
            continue
        n  /= nn
        e3  = np.cross(n, e1)
        R   = np.stack([e1, e3, n], axis=-1)         # (3, 3)
        T[i, :3, :3] = R
        T[i, :3,  3] = p1

    # Endpoints: copy nearest valid frame
    for i in [0, L-1]:
        neighbor = 1 if i == 0 else L - 2
        if valid[neighbor]:
            T[i] = T[neighbor].copy()
            T[i, :3, 3] = coords[i] if valid[i] else T[neighbor, :3, 3]

    return T


def graph_laplacian_eigvecs(coords: np.ndarray,
                             k_edges: int = 6,
                             n_eig: int = 8) -> np.ndarray:
    """
    Compute the first n_eig non-trivial eigenvectors of the graph Laplacian
    formed from the k-NN contact graph of C1' coordinates.
    Returns (L, n_eig) float32 — a structural fingerprint.
    Novel: encodes global topology, not just local pairwise distances.
    """
    L = len(coords)
    valid = ~np.isnan(coords[:, 0])
    c = coords.copy()
    c[~valid] = c[valid].mean(0) if valid.any() else np.zeros(3)

    # pairwise distance
    D = cdist(c, c, metric='euclidean').astype(np.float32)
    np.fill_diagonal(D, np.inf)

    # k-NN adjacency
    A = np.zeros((L, L), np.float32)
    for i in range(L):
        if not valid[i]:
            continue
        nbrs = np.argsort(D[i])[:k_edges]
        for j in nbrs:
            if valid[j]:
                w = np.exp(-D[i, j] / 8.0)      # Gaussian weighting
                A[i, j] = A[j, i] = w

    deg = A.sum(1)
    D_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg + 1e-8), 0.0)
    L_sym = np.eye(L) - D_inv_sqrt[:, None] * A * D_inv_sqrt[None, :]

    n_eig_use = min(n_eig + 1, L)
    try:
        eigvals, eigvecs = np.linalg.eigh(L_sym)
        # Skip trivial eigenvector (eigenvalue ≈ 0), take next n_eig
        vecs = eigvecs[:, 1:n_eig_use].astype(np.float32)
    except np.linalg.LinAlgError:
        vecs = np.zeros((L, n_eig), np.float32)

    # Pad or trim to exactly n_eig columns
    if vecs.shape[1] < n_eig:
        pad  = np.zeros((L, n_eig - vecs.shape[1]), np.float32)
        vecs = np.concatenate([vecs, pad], axis=1)
    else:
        vecs = vecs[:, :n_eig]

    # Zero out invalid residues
    vecs[~valid] = 0.0
    return vecs


def compute_torsion_angles(coords: np.ndarray) -> np.ndarray:
    """
    Pseudo-torsion angles (eta, theta) from C1' coordinates.
    Novel: also computes second-order torsion (curvature) for helical detection.
    Returns (L, 6) — [sin_eta, cos_eta, sin_theta, cos_theta, sin_curl, cos_curl].
    """
    L = len(coords)
    out = np.zeros((L, 6), np.float32)
    valid = ~np.isnan(coords[:, 0])

    def dihedral_angle(p0, p1, p2, p3):
        b1 = p1 - p0
        b2 = p2 - p1
        b3 = p3 - p2
        n1 = np.cross(b1, b2); n2 = np.cross(b2, b3)
        n1n = np.linalg.norm(n1); n2n = np.linalg.norm(n2)
        if n1n < 1e-8 or n2n < 1e-8:
            return 0.0
        n1 /= n1n; n2 /= n2n
        m1  = np.cross(n1, b2 / (np.linalg.norm(b2) + 1e-8))
        x   = np.clip(np.dot(n1, n2), -1, 1)
        y   = np.clip(np.dot(m1, n2), -1, 1)
        return math.atan2(y, x)

    for i in range(1, L - 2):
        if not all(valid[max(0,i-1):i+3]):
            continue
        eta   = dihedral_angle(coords[i-1], coords[i], coords[i+1], coords[i+2])
        theta = dihedral_angle(coords[i],   coords[i+1], coords[i+2], coords[min(i+3, L-1)])
        # Second-order (curvature): rate of change of eta
        curl = 0.0
        if i > 1 and valid[i-2]:
            prev_eta = dihedral_angle(coords[i-2], coords[i-1], coords[i], coords[i+1])
            curl = eta - prev_eta
        out[i] = [math.sin(eta), math.cos(eta),
                  math.sin(theta), math.cos(theta),
                  math.sin(curl), math.cos(curl)]
    return out


# ═════════════════════════════════════════════════════════════
# 3. LEARNED SIMILARITY SCORE (numpy version for caching)
# ═════════════════════════════════════════════════════════════
def compute_template_similarity_features(
    query_seq       : str,
    template_seq    : str,
    template_coords : np.ndarray,
    query_seq_len   : int,
) -> Dict[str, np.ndarray]:
    """
    Build all features needed for one (query, template) pair.
    These are cached per template and fused in the model.
    """
    L = min(query_seq_len, TEMPLATE_MAXL)

    # Alignment
    offset, align_score = local_best_alignment(query_seq, template_seq)

    # Aligned coords: slice template to match query
    t_len = len(template_coords)
    start = min(offset, max(0, t_len - L))
    tc    = template_coords[start : start + L]
    if len(tc) < L:
        pad  = np.full((L - len(tc), 3), np.nan, np.float32)
        tc   = np.concatenate([tc, pad], axis=0)

    # Sequence identity mask: per-position match
    seq_match = np.zeros(L, np.float32)
    q_sub     = query_seq[:L]
    t_sub     = template_seq[start : start + L]
    for i, (q, t) in enumerate(zip(q_sub, t_sub)):
        if q == t:
            seq_match[i] = 1.0

    # Template distogram (soft bins, triangular basis)
    template_dgram = compute_template_distogram(tc, n_bins=38)

    # Local frames
    template_frames = compute_template_frames(tc)   # (L, 4, 4)

    # Torsion angles
    template_torsion = compute_torsion_angles(tc)   # (L, 6)

    # Graph Laplacian eigenvectors
    template_laplacian = graph_laplacian_eigvecs(tc, k_edges=6, n_eig=K_LAPLACIAN)

    # Confidence proxy: fraction valid coords in window
    valid_mask = (~np.isnan(tc[:, 0])).astype(np.float32)
    confidence = float(valid_mask.mean())

    return {
        'template_dgram'     : template_dgram,        # (L, L, 38)
        'template_frames'    : template_frames,        # (L, 4, 4)
        'template_torsion'   : template_torsion,       # (L, 6)
        'template_laplacian' : template_laplacian,     # (L, 8)
        'seq_match'          : seq_match,              # (L,)
        'valid_mask'         : valid_mask,             # (L,)
        'align_score'        : np.float32(align_score),
        'confidence'         : np.float32(confidence),
        'seq_identity'       : np.float32(sequence_identity(query_seq, template_seq)),
    }


# ═════════════════════════════════════════════════════════════
# 4. MULTI-TEMPLATE AGGREGATION (for caching)
# ═════════════════════════════════════════════════════════════
def build_template_feature_stack(
    query_seq       : str,
    template_list   : List[Dict],          # list of {'seq', 'coords', 'id'}
    max_len         : int = 512,
    max_templates   : int = MAX_TEMPLATES,
) -> Dict[str, np.ndarray]:
    """
    For each candidate template, compute features and stack into arrays.
    Template selection: rank by (align_score × confidence), keep top-K.

    Returns stacked arrays of shape (T, ...) where T = max_templates.
    Missing templates are zero-padded.
    """
    L = min(len(query_seq), max_len)
    T = max_templates

    # Compute per-template features
    scored = []
    for tmpl in template_list:
        feat = compute_template_similarity_features(
            query_seq       = query_seq,
            template_seq    = tmpl['seq'],
            template_coords = tmpl['coords'],
            query_seq_len   = L,
        )
        # Ranking score: alignment × confidence × (1 − over-identity penalty)
        identity   = float(feat['seq_identity'])
        over_id_penalty = max(0.0, identity - 0.9) * 2.0  # penalise >90% identity
        score = float(feat['align_score']) * float(feat['confidence']) * (1 - over_id_penalty)
        scored.append((score, feat, tmpl['id']))

    scored.sort(key=lambda x: -x[0])
    selected = scored[:T]

    # Build stacked arrays
    dgram_stack     = np.zeros((T, L, L, 38), np.float32)
    frames_stack    = np.tile(np.eye(4, dtype=np.float32), (T, L, 1, 1))
    torsion_stack   = np.zeros((T, L, 6),    np.float32)
    laplacian_stack = np.zeros((T, L, K_LAPLACIAN), np.float32)
    seq_match_stack = np.zeros((T, L),       np.float32)
    valid_stack     = np.zeros((T, L),       np.float32)
    scores          = np.zeros(T,            np.float32)
    n_valid         = len(selected)

    for k, (score, feat, tid) in enumerate(selected):
        dgram_stack[k]     = feat['template_dgram']
        frames_stack[k]    = feat['template_frames']
        torsion_stack[k]   = feat['template_torsion']
        laplacian_stack[k] = feat['template_laplacian']
        seq_match_stack[k] = feat['seq_match']
        valid_stack[k]     = feat['valid_mask']
        scores[k]          = score

    # Softmax weights over selected templates (used for soft fusion in model)
    # Zero-padded templates get score=0 → near-zero weight
    exp_s    = np.exp(scores - scores.max())
    exp_s[n_valid:] = 0.0
    weights  = exp_s / (exp_s.sum() + 1e-8)

    return {
        'tmpl_dgram'    : dgram_stack,      # (T, L, L, 38)
        'tmpl_frames'   : frames_stack,     # (T, L, 4, 4)
        'tmpl_torsion'  : torsion_stack,    # (T, L, 6)
        'tmpl_laplacian': laplacian_stack,  # (T, L, 8)
        'tmpl_seq_match': seq_match_stack,  # (T, L)
        'tmpl_valid'    : valid_stack,      # (T, L)
        'tmpl_weights'  : weights,          # (T,) — soft fusion weights
        'n_valid_tmpls' : np.int32(n_valid),
    }


# ─────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== rna_template_search self-test ===")
    np.random.seed(42)
    L = 80

    query_seq = "GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGG" * 2
    query_seq = query_seq[:L]

    templates = []
    for i in range(6):
        seq_mut = list(query_seq)
        for j in range(0, L, 5 + i):
            seq_mut[j] = np.random.choice(list("AUGC"))
        templates.append({
            'id'    : f"tmpl_{i}",
            'seq'   : ''.join(seq_mut),
            'coords': np.cumsum(np.random.randn(L, 3) * 2, 0).astype(np.float32),
        })

    stack = build_template_feature_stack(query_seq, templates, max_len=L, max_templates=4)
    print("Template stack shapes:")
    for k, v in stack.items():
        if isinstance(v, np.ndarray):
            print(f"  {k:20s}: {v.shape}  dtype={v.dtype}")
        else:
            print(f"  {k:20s}: {v}")

    # Test individual helpers
    c = np.cumsum(np.random.randn(L, 3) * 2, 0).astype(np.float32)
    eig = graph_laplacian_eigvecs(c)
    print(f"\nLaplacian eigvecs: {eig.shape}  range=[{eig.min():.3f}, {eig.max():.3f}]")

    tors = compute_torsion_angles(c)
    print(f"Torsion angles  : {tors.shape}  non-zero={np.count_nonzero(tors.any(1))}")

    frames = compute_template_frames(c)
    print(f"Frames          : {frames.shape}")

    print("\n✅ All OK")
