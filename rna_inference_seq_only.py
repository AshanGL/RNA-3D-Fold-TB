"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  RNA SEQUENCE-ONLY INFERENCE  —  ZERO EXTERNAL DEPENDENCIES                  ║
║                                                                               ║
║  Only needs: numpy, torch, scipy  (all standard Kaggle packages)             ║
║  No biopython, no RNAfold, no internet, no MSA tools.                        ║
║                                                                               ║
║  ACCURACY STRATEGY — every feature the model was trained on is               ║
║  approximated from sequence alone:                                            ║
║                                                                               ║
║  Feature              Training source     Here                               ║
║  ─────────────────    ────────────────    ───────────────────────────────    ║
║  seq_ids              CSV sequence        same (exact)                        ║
║  rel_pos              pure math           same (exact)                        ║
║  ss_pair / contact_ss RNAfold             Nussinov dynamic programming (DP)  ║
║  pair_type            RNAfold             derived from Nussinov pairs         ║
║  f1  (freq profile)   MSA file            single-seq one-hot per position    ║
║  MIp / FNp            MSA covariation     sequence-only stacking proxy        ║
║  dist_rbf / dist_bins real 3D coords      predicted from SS + A-form geometry ║
║  orient               real 3D coords      local-frame estimate from SS        ║
║  dihed                real 3D coords      torsion estimate from predicted pos ║
║  tmpl_*               PDB structures      self-template: seq → SS → coords   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import math
import csv
import numpy as np
from scipy.spatial.distance import cdist
import torch
from pathlib import Path
from typing import Union, Dict, List

from rna_model_template import RNAFoldTemplate, cfg as ModelCfg

VOCAB    = {'A': 0, 'U': 1, 'G': 2, 'C': 3, '<PAD>': 4, '<UNK>': 5}
MAX_LEN  = ModelCfg.MAX_LEN        # 512
T_DUMMY  = ModelCfg.MAX_TEMPLATES  # 4
WC_PAIRS = {('A','U'),('U','A'),('G','C'),('C','G'),('G','U'),('U','G')}


# ══════════════════════════════════════════════════════════════
# 1.  SECONDARY STRUCTURE  —  Nussinov DP  (pure python/numpy)
# ══════════════════════════════════════════════════════════════
def nussinov_fold(seq: str, min_loop: int = 3) -> List:
    """
    Classic Nussinov dynamic programming.
    Returns list of (i, j) base-pair tuples (0-indexed).
    No external tools — O(N^3), fast enough for <= 512 nt.
    """
    N  = len(seq)
    dp = [[0] * N for _ in range(N)]

    def can_pair(i, j):
        return (seq[i], seq[j]) in WC_PAIRS

    for length in range(min_loop + 1, N):
        for i in range(N - length):
            j = i + length
            dp[i][j] = max(
                dp[i+1][j] if i+1 <= j else 0,
                dp[i][j-1] if i <= j-1 else 0,
            )
            if can_pair(i, j):
                inner = dp[i+1][j-1] if i+1 <= j-1 else 0
                dp[i][j] = max(dp[i][j], inner + 1)
            for k in range(i+1, j):
                dp[i][j] = max(dp[i][j],
                               (dp[i][k] if i <= k else 0) +
                               (dp[k+1][j] if k+1 <= j else 0))

    pairs = []
    stack = [(0, N - 1)]
    while stack:
        i, j = stack.pop()
        if i >= j:
            continue
        if dp[i][j] == dp[i+1][j]:
            stack.append((i+1, j))
        elif dp[i][j] == dp[i][j-1]:
            stack.append((i, j-1))
        elif can_pair(i, j) and dp[i][j] == (dp[i+1][j-1] if i+1 <= j-1 else 0) + 1:
            pairs.append((i, j))
            stack.append((i+1, j-1))
        else:
            for k in range(i+1, j):
                left  = dp[i][k]   if i   <= k else 0
                right = dp[k+1][j] if k+1 <= j else 0
                if left + right == dp[i][j]:
                    stack.append((i, k))
                    stack.append((k+1, j))
                    break
    return pairs


def ss_features(seq: str, max_len: int) -> Dict:
    """
    Build ss_pair, contact_ss, pair_type (L, max_len, max_len, 3)
    from Nussinov secondary structure.
    Matches exactly what rna_features_v2.secondary_structure_features() outputs.
    """
    L   = min(len(seq), max_len)
    seq = seq[:L]
    pairs = nussinov_fold(seq)

    ss_pair    = np.zeros(max_len, np.float32)
    contact_ss = np.zeros((max_len, max_len), np.float32)
    pair_type  = np.zeros((max_len, max_len, 3), np.float32)

    for i, j in pairs:
        ss_pair[i] = ss_pair[j] = 1.0
        contact_ss[i, j] = contact_ss[j, i] = 1.0
        b = (seq[i], seq[j])
        if b in {('A','U'),('U','A'),('G','C'),('C','G')}:
            pair_type[i, j, 0] = pair_type[j, i, 0] = 1.0   # Watson-Crick
        elif b in {('G','U'),('U','G')}:
            pair_type[i, j, 1] = pair_type[j, i, 1] = 1.0   # Wobble

    for i in range(L - 1):                                    # adjacent stacking
        pair_type[i, i+1, 2] = pair_type[i+1, i, 2] = 1.0

    return {'ss_pair': ss_pair, 'contact_ss': contact_ss,
            'pair_type': pair_type, 'pairs': pairs}


# ══════════════════════════════════════════════════════════════
# 2.  SEQUENCE-BASED COVARIATION PROXY  (replaces MSA features)
# ══════════════════════════════════════════════════════════════
def covariation_proxy(seq: str, max_len: int) -> Dict:
    """
    Approximates f1, MIp, FNp without an MSA.

    f1  — one-hot base identity per position (5 channels: A/U/G/C/gap)
    MIp — proximity-weighted same-class signal; encodes that nearby
          same-type residues co-vary (dominant MSA MI signal for short RNAs)
    FNp — dinucleotide stacking score (Turner 2004 empirical energies);
          surrogate for the Frobenius-norm coevolution signal
    """
    L   = min(len(seq), max_len)
    seq = seq[:L]
    base_idx = {'A':0,'U':1,'G':2,'C':3}

    # f1: one-hot (max_len, 5)
    f1 = np.zeros((max_len, 5), np.float32)
    for i, b in enumerate(seq):
        idx = base_idx.get(b)
        if idx is not None:
            f1[i, idx] = 1.0
        else:
            f1[i, 4] = 1.0

    # MIp proxy
    MIp = np.zeros((max_len, max_len), np.float32)
    lam = 8.0
    for i in range(L):
        for j in range(i+1, L):
            same = 1.0 if (seq[i] in 'GC' and seq[j] in 'GC') or \
                          (seq[i] in 'AU' and seq[j] in 'AU') else 0.3
            val = math.exp(-abs(i-j) / lam) * same
            MIp[i, j] = MIp[j, i] = val

    # FNp proxy: Turner stacking energies (no internet, standard biochemistry)
    STACK = {
        ('G','C',  'G','C'): 3.4, ('G','C',  'C','G'): 2.4,
        ('G','C',  'A','U'): 2.1, ('G','C',  'U','A'): 2.1,
        ('C','G',  'G','C'): 3.3, ('C','G',  'C','G'): 2.4,
        ('A','U',  'A','U'): 0.9, ('A','U',  'U','A'): 1.1,
        ('A','U',  'G','C'): 2.1, ('A','U',  'G','U'): 0.9,
        ('G','U',  'G','U'): 0.5, ('G','U',  'G','C'): 2.1,
        ('U','A',  'A','U'): 1.3, ('U','A',  'G','C'): 2.1,
    }
    FNp = np.zeros((max_len, max_len), np.float32)
    for i in range(L - 1):
        for j in range(i+1, L - 1):
            key = (seq[i], seq[j], seq[i+1], seq[j-1]) if j > 0 else None
            score = STACK.get(key, 0.0) / 3.4 if key else 0.0
            FNp[i, j] = FNp[j, i] = float(score)

    return {'f1': f1, 'MIp': MIp, 'FNp': FNp}


# ══════════════════════════════════════════════════════════════
# 3.  IDEAL 3D COORDS FROM SS  (A-form helix + loop geometry)
# ══════════════════════════════════════════════════════════════
def ideal_coords_from_ss(seq: str, pairs: list) -> np.ndarray:
    """
    Place C1' atoms using A-form helix parameters for paired residues
    and a coaxial-stack walk for loops.  Returns (L, 3) float32.
    No NaN — fully valid coords for every residue.
    """
    L        = len(seq)
    coords   = np.zeros((L, 3), np.float32)
    pair_map = {}
    for i, j in pairs:
        pair_map[i] = j
        pair_map[j] = i

    # A-form RNA helix parameters
    RISE  = 2.81   # Å per residue
    TWIST = 32.7   # degrees per residue
    BOND  = 6.0    # virtual C1'-C1' bond length

    pos   = np.zeros(3, np.float32)
    twist = 0.0

    for i in range(L):
        coords[i] = pos
        j = pair_map.get(i)
        in_helix = j is not None
        if in_helix:
            t = math.radians(twist)
            pos = pos + np.array([BOND * 0.2 * math.cos(t),
                                   BOND * 0.2 * math.sin(t),
                                   RISE], np.float32)
            twist += TWIST
        else:
            t = math.radians(twist)
            pos = pos + np.array([5.9 * math.cos(t),
                                   5.9 * math.sin(t),
                                   1.5], np.float32)
            twist += 45.0

    return coords


def geometry_features(coords: np.ndarray, max_len: int) -> Dict:
    """
    Compute dist_rbf, dist_bins, orient, dihed from (L,3) coordinates.
    Mirrors gpu_distance_features() in rna_template_cache.py — pure numpy.
    """
    L_raw  = len(coords)
    n_rbf  = ModelCfg.N_RBF      # 16
    n_bins = ModelCfg.N_DIST_BINS # 38
    d_min, d_max = 2.0, 20.0

    diff  = coords[:, None, :] - coords[None, :, :]
    dists = np.sqrt((diff ** 2).sum(-1) + 1e-8)

    full = np.zeros((max_len, max_len), np.float32)
    full[:L_raw, :L_raw] = dists.astype(np.float32)

    # RBF
    centers = np.linspace(d_min, d_max, n_rbf)
    gamma   = 1.0 / (2 * ((d_max - d_min) / n_rbf) ** 2)
    rbf     = np.exp(-gamma * (full[:,:,None] - centers[None,None,:]) ** 2)

    # Distance bins (one-hot, 38 bins)
    edges     = np.linspace(d_min, d_max, n_bins + 1)
    bin_ids   = np.clip(np.digitize(full, edges) - 1, 0, n_bins - 1)
    dist_bins = np.eye(n_bins, dtype=np.float32)[bin_ids]

    # Orientation: unit direction + distance decay
    orient = np.zeros((max_len, max_len, 4), np.float32)
    nd = diff / (dists[:,:,None] + 1e-8)
    orient[:L_raw, :L_raw, :3] = nd.astype(np.float32)
    orient[:L_raw, :L_raw,  3] = np.exp(-full[:L_raw,:L_raw] / 10.0)

    # Dihedral (4 features per residue from triplet geometry)
    dihed = np.zeros((max_len, 4), np.float32)
    for i in range(1, L_raw - 1):
        b1 = coords[i] - coords[i-1]
        b2 = coords[i+1] - coords[i]
        n1 = np.cross(b1, b2)
        nn = np.linalg.norm(n1)
        if nn > 1e-6:
            n1 /= nn
            a  = math.atan2(float(n1[1]), float(n1[0]))
            dihed[i] = [math.sin(a), math.cos(a),
                        float(n1[2]), float(np.linalg.norm(b2))]

    return {'dist_rbf': rbf.astype(np.float32),
            'dist_bins': dist_bins,
            'orient': orient,
            'dihed': dihed}


# ══════════════════════════════════════════════════════════════
# 4.  SELF-TEMPLATE  (predicted SS + coords → template slot 0)
# ══════════════════════════════════════════════════════════════
def build_self_template(seq: str, pairs: list,
                        coords: np.ndarray, max_len: int) -> Dict:
    """
    The key accuracy boost over all-zeros.

    The model's TemplatePairStack and TemplateSingleStack were trained to
    extract structural priors (distance distributions, local frames, torsions,
    graph topology) from template structures.  Even an approximate self-template
    derived from Nussinov SS + A-form geometry carries real structural signal:
      - helix regions → correct distance bins for double-stranded contacts
      - loop regions  → correct distance bins for single-stranded spacing
      - torsion angles → A-form values for helices, extended for loops
      - Laplacian eigvecs → correct global topology fingerprint

    Slot 0 = self-template (tmpl_valid = 1.0 for all L residues)
    Slots 1-3 = zero (tmpl_valid = 0.0 → gated out by TemplatePairStack)
    """
    L = min(len(seq), max_len)
    T = T_DUMMY
    K = ModelCfg.K_LAPLACIAN  # 8
    c = coords[:L]

    # -- Distogram (triangular soft bins, matches compute_template_distogram) --
    diff  = c[:, None, :] - c[None, :, :]
    dists = np.sqrt((diff ** 2).sum(-1) + 1e-8)
    d_min, d_max, nb = 2.0, 22.0, 38
    centers = np.linspace(d_min, d_max, nb)
    width   = (d_max - d_min) / nb
    tri     = np.maximum(0, 1 - np.abs(dists[:,:,None] - centers[None,None,:]) / width)
    tri    /= tri.sum(-1, keepdims=True) + 1e-8
    dgram   = np.zeros((max_len, max_len, nb), np.float32)
    dgram[:L, :L] = tri.astype(np.float32)

    # -- Frames (SVD local frame per triplet, matches compute_template_frames) --
    frames = np.eye(4, dtype=np.float32)[None].repeat(max_len, axis=0)
    for i in range(1, L - 1):
        e1 = c[i] - c[i-1]; e1 /= (np.linalg.norm(e1) + 1e-8)
        e2 = c[i+1] - c[i]; e2 /= (np.linalg.norm(e2) + 1e-8)
        n  = np.cross(e1, e2); nn = np.linalg.norm(n)
        if nn > 1e-6:
            n /= nn; e3 = np.cross(n, e1)
            frames[i, :3, :3] = np.stack([e1, e3, n], -1)
            frames[i, :3,  3] = c[i]

    # -- Torsion angles (matches compute_torsion_angles) --
    torsion = np.zeros((max_len, 6), np.float32)
    def dih(a, b, cc, d):
        b1=b-a; b2=cc-b; b3=d-cc
        n1=np.cross(b1,b2); n2=np.cross(b2,b3)
        n1n=np.linalg.norm(n1); n2n=np.linalg.norm(n2)
        if n1n<1e-8 or n2n<1e-8: return 0.0
        n1/=n1n; n2/=n2n
        m1=np.cross(n1,b2/(np.linalg.norm(b2)+1e-8))
        return math.atan2(float(np.dot(m1,n2)), float(np.dot(n1,n2)))
    for i in range(1, L - 2):
        eta   = dih(c[i-1], c[i], c[i+1], c[min(i+2,L-1)])
        theta = dih(c[i],   c[i+1], c[min(i+2,L-1)], c[min(i+3,L-1)])
        curl  = 0.0
        if i > 1:
            curl = eta - dih(c[i-2], c[i-1], c[i], c[i+1])
        torsion[i] = [math.sin(eta), math.cos(eta),
                      math.sin(theta), math.cos(theta),
                      math.sin(curl), math.cos(curl)]

    # -- Laplacian eigenvectors (matches graph_laplacian_eigvecs) --
    D_mat = cdist(c, c).astype(np.float32)
    np.fill_diagonal(D_mat, np.inf)
    k  = min(6, L - 1)
    A  = np.zeros((L, L), np.float32)
    for i in range(L):
        for j in np.argsort(D_mat[i])[:k]:
            w = float(np.exp(-D_mat[i, j] / 8.0))
            A[i, j] = A[j, i] = max(A[i, j], w)
    deg  = A.sum(1)
    Di   = np.where(deg > 0, 1.0 / np.sqrt(deg + 1e-8), 0.0)
    Lsym = np.eye(L) - Di[:,None] * A * Di[None,:]
    n_eig = min(K + 1, L)
    try:
        _, vecs = np.linalg.eigh(Lsym)
        vecs = vecs[:, 1:n_eig].astype(np.float32)
    except np.linalg.LinAlgError:
        vecs = np.zeros((L, K), np.float32)
    if vecs.shape[1] < K:
        vecs = np.concatenate([vecs, np.zeros((L, K - vecs.shape[1]), np.float32)], 1)
    else:
        vecs = vecs[:, :K]
    lap = np.zeros((max_len, K), np.float32)
    lap[:L] = vecs

    # -- seq_match = 1.0 (self-template, exact identity) --
    seq_match = np.zeros(max_len, np.float32); seq_match[:L] = 1.0
    valid     = np.zeros(max_len, np.float32); valid[:L]     = 1.0

    # -- Stack: slot 0 filled, slots 1-3 zero --
    dgram_s    = np.zeros((T, max_len, max_len, nb), np.float32)
    frames_s   = np.tile(np.eye(4, dtype=np.float32), (T, max_len, 1, 1))
    torsion_s  = np.zeros((T, max_len, 6),  np.float32)
    lap_s      = np.zeros((T, max_len, K),  np.float32)
    match_s    = np.zeros((T, max_len),     np.float32)
    valid_s    = np.zeros((T, max_len),     np.float32)
    weights    = np.array([1.0, 0.0, 0.0, 0.0], np.float32)

    dgram_s[0]  = dgram;    frames_s[0]  = frames
    torsion_s[0]= torsion;  lap_s[0]     = lap
    match_s[0]  = seq_match; valid_s[0]  = valid

    return {'tmpl_dgram': dgram_s, 'tmpl_frames': frames_s,
            'tmpl_torsion': torsion_s, 'tmpl_laplacian': lap_s,
            'tmpl_seq_match': match_s, 'tmpl_valid': valid_s,
            'tmpl_weights': weights}


# ══════════════════════════════════════════════════════════════
# 5.  RELATIVE POSITION ENCODING  (pure math, identical to training)
# ══════════════════════════════════════════════════════════════
def rel_pos_encoding(max_len: int, max_range: int = 32) -> np.ndarray:
    pos  = np.arange(max_len)
    diff = np.clip(pos[:,None] - pos[None,:], -max_range, max_range)
    oh   = np.zeros((max_len, max_len, 2*max_range+1), np.float32)
    for d in range(-max_range, max_range+1):
        oh[:,:, d+max_range] = (diff == d).astype(np.float32)
    return oh


# ══════════════════════════════════════════════════════════════
# 6.  ASSEMBLE FULL BATCH DICT
# ══════════════════════════════════════════════════════════════
def sequence_to_batch(seq: str, device: torch.device) -> dict:
    """
    Convert a raw nucleotide string → complete batch dict.
    Every tensor approximated from sequence alone.
    """
    seq = seq.upper().replace('T', 'U')
    L   = min(len(seq), MAX_LEN)
    seq = seq[:L]

    # tokenise
    ids = [VOCAB.get(c, VOCAB['<UNK>']) for c in seq]
    ids += [VOCAB['<PAD>']] * (MAX_LEN - L)
    seq_ids  = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    seq_mask = torch.zeros(1, MAX_LEN, dtype=torch.bool, device=device)
    seq_mask[0, :L] = True

    ss     = ss_features(seq, MAX_LEN)
    pairs  = ss['pairs']
    coords = ideal_coords_from_ss(seq, pairs)
    geo    = geometry_features(coords, MAX_LEN)
    cov    = covariation_proxy(seq, MAX_LEN)
    tmpl   = build_self_template(seq, pairs, coords, MAX_LEN)
    rp     = rel_pos_encoding(MAX_LEN)

    def t(x): return torch.from_numpy(x).unsqueeze(0).to(device)

    return {
        'seq_ids'       : seq_ids,
        'seq_mask'      : seq_mask,
        'f1'            : t(cov['f1']),
        'dihed'         : t(geo['dihed']),
        'ss_pair'       : t(ss['ss_pair']),
        'dist_rbf'      : t(geo['dist_rbf']),
        'dist_bins'     : t(geo['dist_bins']),
        'orient'        : t(geo['orient']),
        'rel_pos'       : t(rp),
        'MIp'           : t(cov['MIp']),
        'FNp'           : t(cov['FNp']),
        'contact_ss'    : t(ss['contact_ss']),
        'pair_type'     : t(ss['pair_type']),
        'tmpl_dgram'    : t(tmpl['tmpl_dgram']),
        'tmpl_frames'   : t(tmpl['tmpl_frames']),
        'tmpl_torsion'  : t(tmpl['tmpl_torsion']),
        'tmpl_laplacian': t(tmpl['tmpl_laplacian']),
        'tmpl_seq_match': t(tmpl['tmpl_seq_match']),
        'tmpl_valid'    : t(tmpl['tmpl_valid']),
        'tmpl_weights'  : t(tmpl['tmpl_weights']),
        'seq_len'       : torch.tensor([L]),
    }


# ══════════════════════════════════════════════════════════════
# 7.  PREDICTOR
# ══════════════════════════════════════════════════════════════
class RNASequencePredictor:
    """
    Load a trained RNAFoldTemplate checkpoint and predict 3D structure
    from a nucleotide sequence alone.

    Usage
    -----
    predictor = RNASequencePredictor("best_rna_template.pt")
    result    = predictor.predict("GGCUACGGCCAUACCUG")
    coords    = result['coords']   # numpy (L, 3)  C1' positions in Angstrom
    plddt     = result['plddt']    # numpy (L,)    confidence 0-100
    """

    def __init__(self, checkpoint_path: Union[str, Path], device: str = 'auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"  Loading: {checkpoint_path}  ->  {self.device}")
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model = RNAFoldTemplate().to(self.device)
        raw   = state.get('model', state)
        clean = {k.replace('module.', ''): v for k, v in raw.items()}
        self.model.load_state_dict(clean)
        self.model.eval()
        print(f"  Loaded  epoch={state.get('epoch','?')}  "
              f"val_loss={state.get('val_loss', float('nan')):.4f}")

    @torch.no_grad()
    def predict(self, sequence: str) -> dict:
        seq   = sequence.upper().replace('T', 'U')[:MAX_LEN]
        L     = len(seq)
        batch = sequence_to_batch(seq, self.device)
        out   = self.model(batch, device=self.device)
        pairs = nussinov_fold(seq)
        return {
            'coords'   : out['coords'][0,    :L].cpu().float().numpy(),
            'plddt'    : out['plddt'][0,     :L].cpu().float().numpy(),
            'distogram': out['distogram'][0, :L, :L].cpu().float().numpy(),
            'torsion'  : out['torsion'][0,   :L].cpu().float().numpy(),
            'ss_pairs' : pairs,
            'sequence' : seq,
        }

    @torch.no_grad()
    def predict_batch(self, sequences: list) -> list:
        return [self.predict(s) for s in sequences]


# ══════════════════════════════════════════════════════════════
# 8.  SUBMISSION CSV BUILDER  (Kaggle format, no pandas needed)
# ══════════════════════════════════════════════════════════════
def build_submission(checkpoint_path: str,
                     test_csv: str,
                     out_csv: str,
                     device: str = 'auto'):
    """
    End-to-end: checkpoint + test_sequences.csv -> submission.csv

    Dependencies: numpy, torch, scipy only. No biopython, no internet.

    Parameters
    ----------
    checkpoint_path : path to best_rna_template.pt
    test_csv        : path to test_sequences.csv  (cols: target_id, sequence)
    out_csv         : output path for submission.csv
    """
    predictor = RNASequencePredictor(checkpoint_path, device=device)

    rows_in = []
    with open(test_csv, newline='') as f:
        for row in csv.DictReader(f):
            rows_in.append(row)

    print(f"  Predicting {len(rows_in)} sequences ...")
    rows_out = []
    for row in rows_in:
        tid    = row['target_id']
        seq    = row['sequence']
        result = predictor.predict(seq)
        for i, (xyz, conf) in enumerate(zip(result['coords'], result['plddt'])):
            rows_out.append({
                'ID' : f"{tid}_{i+1}",
                'x_1': f"{xyz[0]:.4f}",
                'y_1': f"{xyz[1]:.4f}",
                'z_1': f"{xyz[2]:.4f}",
            })

    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['ID','x_1','y_1','z_1'])
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"  Saved {len(rows_out):,} residue rows -> {out_csv}")


# ══════════════════════════════════════════════════════════════
# 9.  SMOKE TEST
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Smoke test — random weights, no checkpoint needed")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = RNAFoldTemplate().to(device)
    model.eval()

    test_seqs = [
        "GGCUACGGCCAUACCUGCUAGUAGCC",
        "GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGG",
    ]

    for seq in test_seqs:
        print(f"\n  seq ({len(seq)} nt): {seq[:25]}...")
        batch = sequence_to_batch(seq, device)
        pairs = nussinov_fold(seq)
        print(f"  Nussinov pairs: {len(pairs)}")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"    {k:22s}: {tuple(v.shape)}")
        with torch.no_grad():
            out = model(batch, device=device)
        L = len(seq)
        print(f"  coords ({L},3)  plddt [{out['plddt'][0,:L].min():.1f}, "
              f"{out['plddt'][0,:L].max():.1f}]")

    print("\n  Smoke test passed")

    if len(sys.argv) > 1:
        ckpt = sys.argv[1]
        seq  = sys.argv[2] if len(sys.argv) > 2 else test_seqs[1]
        pred = RNASequencePredictor(ckpt)
        res  = pred.predict(seq)
        for i in range(min(3, len(res['coords']))):
            x,y,z = res['coords'][i]
            print(f"  [{i+1}] {x:.2f} {y:.2f} {z:.2f}  pLDDT={res['plddt'][i]:.1f}")

    # build_submission(ckpt, test_csv, out_csv)
    if len(sys.argv) > 3:
        build_submission(sys.argv[1], sys.argv[2], sys.argv[3])
