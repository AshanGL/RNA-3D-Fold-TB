"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  RNA TEMPLATE FEATURE CACHE                                                   ║
║                                                                               ║
║  Extends rna_feature_cache.py (v3) with template stack features.             ║
║  All v3 sequence features are retained and extended.                          ║
║                                                                               ║
║  Cache layout per sample:                                                     ║
║   /kaggle/working/feat_cache/{split}/{target_id}.npz                          ║
║     — all v3 features (seq, MSA, SS, geometry, rel_pos)                      ║
║     — tmpl_dgram     : (T, L, L, 38)                                          ║
║     — tmpl_frames    : (T, L, 4, 4)                                           ║
║     — tmpl_torsion   : (T, L, 6)                                              ║
║     — tmpl_laplacian : (T, L, 8)                                              ║
║     — tmpl_seq_match : (T, L)                                                 ║
║     — tmpl_valid     : (T, L)                                                 ║
║     — tmpl_weights   : (T,)                                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os, gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# v3 feature imports
from rna_features_v2 import (
    find_msa_file, msa_covariation_features,
    secondary_structure_features, geometric_features,
    relative_position_encoding,
)
from rna_template_search import build_template_feature_stack

CACHE_VERSION = "tmpl_v1"
MAX_TEMPLATES = 4


# ─────────────────────────────────────────────────────────────
# GPU-accelerated distance & RBF (from v3 cache)
# ─────────────────────────────────────────────────────────────
def gpu_distance_features(coords_list: List[np.ndarray],
                           max_len: int,
                           device: torch.device,
                           n_rbf: int = 16,
                           d_min: float = 2.0,
                           d_max: float = 20.0) -> List[Dict]:
    """Batch pairwise distances + RBF on GPU."""
    results = []
    for coords in coords_list:
        if coords is None:
            results.append({
                'dist_norm': np.zeros((max_len, max_len), np.float32),
                'dist_rbf' : np.zeros((max_len, max_len, n_rbf), np.float32),
                'dist_bins': np.zeros((max_len, max_len, 38), np.float32),
                'contact_3d': np.zeros((max_len, max_len), np.float32),
                'orient'   : np.zeros((max_len, max_len, 4), np.float32),
                'dihed'    : np.zeros((max_len, 4), np.float32),
                'valid_mask': np.zeros(max_len, np.float32),
            })
            continue

        L_raw = min(len(coords), max_len)
        c     = coords[:L_raw].copy()
        valid = ~np.isnan(c[:, 0])
        c[~valid] = 0.0

        ct = torch.tensor(c, device=device)
        vt = torch.tensor(valid, device=device)
        ct[~vt] = 0.0

        diff  = ct.unsqueeze(0) - ct.unsqueeze(1)
        dists = diff.norm(dim=-1)
        dists = dists * vt.float().unsqueeze(0) * vt.float().unsqueeze(1)
        dists_np = dists.cpu().numpy().astype(np.float32)

        full_dist = np.zeros((max_len, max_len), np.float32)
        full_dist[:L_raw, :L_raw] = dists_np
        mx = full_dist.max()
        dist_norm = full_dist / (mx + 1e-8)

        centers = np.linspace(d_min, d_max, n_rbf)
        gamma   = 1.0 / (2 * ((d_max - d_min) / n_rbf) ** 2)
        rbf     = np.exp(-gamma * (full_dist[:, :, None] - centers[None, None, :]) ** 2)

        edges     = np.linspace(d_min, d_max, 39)
        bin_ids   = np.digitize(full_dist, edges) - 1
        bin_ids   = np.clip(bin_ids, 0, 37).astype(np.int32)
        dist_bins = np.eye(38, dtype=np.float32)[bin_ids]

        contact   = (full_dist > 0) & (full_dist < 8.0)
        np.fill_diagonal(contact, False)

        valid_full = np.zeros(max_len, np.float32)
        valid_full[:L_raw] = valid.astype(np.float32)

        results.append({
            'dist_norm' : dist_norm,
            'dist_rbf'  : rbf.astype(np.float32),
            'dist_bins' : dist_bins,
            'contact_3d': contact.astype(np.float32),
            'orient'    : np.zeros((max_len, max_len, 4), np.float32),
            'dihed'     : np.zeros((max_len, 4), np.float32),
            'valid_mask': valid_full,
        })
        del ct, vt, diff, dists
    return results


# ─────────────────────────────────────────────────────────────
# SINGLETON RELATIVE POSITION ENCODING
# ─────────────────────────────────────────────────────────────
_rel_pos_cache: Dict[int, np.ndarray] = {}

def get_rel_pos(max_len: int, max_range: int = 32) -> np.ndarray:
    if max_len not in _rel_pos_cache:
        pos  = np.arange(max_len)
        diff = np.clip(pos[:, None] - pos[None, :], -max_range, max_range)
        oh   = np.zeros((max_len, max_len, 2 * max_range + 1), np.float32)
        for d in range(-max_range, max_range + 1):
            oh[:, :, d + max_range] = (diff == d).astype(np.float32)
        _rel_pos_cache[max_len] = oh
    return _rel_pos_cache[max_len]


# ─────────────────────────────────────────────────────────────
# COMPUTE AND SAVE ONE SAMPLE
# ─────────────────────────────────────────────────────────────
def compute_and_save_features(
    row_dict   : dict,
    cache_dir  : str,
    max_len    : int,
    msa_dir    : str,
    geo_feats  : dict,
    template_list: List[dict],
) -> Path:
    """
    Compute all features for one sample and save as .npz.
    row_dict keys: target_id, sequence, split, coords (can be None)
    geo_feats: pre-computed GPU distance features dict
    template_list: list of {'id', 'seq', 'coords'} dicts
    """
    tid  = row_dict['target_id']
    seq  = row_dict['sequence']
    splt = row_dict['split']
    L    = min(len(seq), max_len)

    cache_path = Path(cache_dir) / splt / f"{tid}.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Check version
    if cache_path.exists():
        try:
            existing = np.load(cache_path, allow_pickle=False)
            if existing.get('cache_version', np.array('')).item() == CACHE_VERSION:
                return cache_path
        except Exception:
            pass

    # ── v3 sequence features ──────────────────────────────
    msa_path = find_msa_file(msa_dir, tid)
    cov      = msa_covariation_features(msa_path, max_len, max_seqs=512)
    ss       = secondary_structure_features(seq[:L], max_len)
    rel_pos  = get_rel_pos(max_len, max_range=32)

    # ── Template features ────────────────────────────────
    tmpl_stack = build_template_feature_stack(
        query_seq     = seq[:L],
        template_list = template_list,
        max_len       = L,
        max_templates = MAX_TEMPLATES,
    )

    # ── Assemble and save ────────────────────────────────
    save_dict = {
        'cache_version': np.array(CACHE_VERSION),
        'seq_len'      : np.int32(L),
        # MSA
        'MIp'          : cov['MIp'],
        'FNp'          : cov['FNp'],
        'f1'           : cov['f1'],
        'neff'         : np.float32(cov['neff']),
        # Secondary structure
        'contact_ss'   : ss['contact_ss'],
        'ss_pair'      : ss['ss_pair'],
        'pair_type'    : ss['pair_type'],
        # Geometry (from GPU batch)
        'dist_norm'    : geo_feats['dist_norm'],
        'dist_rbf'     : geo_feats['dist_rbf'],
        'dist_bins'    : geo_feats['dist_bins'],
        'contact_3d'   : geo_feats['contact_3d'],
        'orient'       : geo_feats['orient'],
        'dihed'        : geo_feats['dihed'],
        'valid_mask'   : geo_feats['valid_mask'],
        # Relative position
        'rel_pos'      : rel_pos,
        # Template stack (all novel)
        'tmpl_dgram'    : tmpl_stack['tmpl_dgram'],
        'tmpl_frames'   : tmpl_stack['tmpl_frames'],
        'tmpl_torsion'  : tmpl_stack['tmpl_torsion'],
        'tmpl_laplacian': tmpl_stack['tmpl_laplacian'],
        'tmpl_seq_match': tmpl_stack['tmpl_seq_match'],
        'tmpl_valid'    : tmpl_stack['tmpl_valid'],
        'tmpl_weights'  : tmpl_stack['tmpl_weights'],
        'n_valid_tmpls' : tmpl_stack['n_valid_tmpls'],
    }
    np.savez_compressed(str(cache_path), **save_dict)
    return cache_path


# ─────────────────────────────────────────────────────────────
# PRECOMPUTE SPLIT
# ─────────────────────────────────────────────────────────────
def precompute_split(
    row_dicts    : List[dict],
    cache_dir    : str,
    max_len      : int,
    msa_dir      : str,
    template_db  : Dict[str, List[dict]],   # {target_id: [template_dicts]}
    device       : torch.device,
    desc         : str = 'Caching',
) -> None:
    """
    Pre-compute and cache features for all samples in a split.
    GPU is used for batched distance computation.
    """
    # Identify which samples need caching
    pending = []
    for rd in row_dicts:
        tid   = rd['target_id']
        splt  = rd['split']
        p     = Path(cache_dir) / splt / f"{tid}.npz"
        if p.exists():
            try:
                existing = np.load(str(p), allow_pickle=False)
                if existing.get('cache_version', np.array('')).item() == CACHE_VERSION:
                    continue
            except Exception:
                pass
        pending.append(rd)

    if not pending:
        print(f"  ✅  {desc}: all {len(row_dicts)} samples already cached (v={CACHE_VERSION})")
        return

    print(f"  ⏳  {desc}: caching {len(pending)}/{len(row_dicts)} samples …")

    # Batch GPU distance computation
    batch_size = 32
    for start in tqdm(range(0, len(pending), batch_size), desc=f"  GPU dist ({desc})"):
        batch = pending[start:start + batch_size]
        coords_batch = [rd.get('coords') for rd in batch]
        geo_batch    = gpu_distance_features(coords_batch, max_len, device)

        for rd, geo in zip(batch, geo_batch):
            tid  = rd['target_id']
            tmpls = template_db.get(tid, [])
            compute_and_save_features(
                row_dict      = rd,
                cache_dir     = cache_dir,
                max_len       = max_len,
                msa_dir       = msa_dir,
                geo_feats     = geo,
                template_list = tmpls,
            )

        del geo_batch
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────
class TemplateRNADataset(Dataset):
    """
    Reads pre-cached .npz feature files.
    Identical to CachedRNADataset but adds template keys.
    """
    def __init__(self, cache_dir: str, split: str, ids: List[str]):
        self.cache_dir = Path(cache_dir)
        self.split     = split
        self.ids       = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict:
        tid  = self.ids[idx]
        path = self.cache_dir / self.split / f"{tid}.npz"
        data = np.load(str(path), allow_pickle=False)

        # Convert sequence to token ids (placeholder — actual tokenisation done by trainer)
        return {
            'target_id' : tid,
            'seq_len'   : int(data['seq_len']),
            # v3 features
            'MIp'       : torch.from_numpy(data['MIp']),
            'FNp'       : torch.from_numpy(data['FNp']),
            'f1'        : torch.from_numpy(data['f1']),
            'contact_ss': torch.from_numpy(data['contact_ss']),
            'ss_pair'   : torch.from_numpy(data['ss_pair']),
            'pair_type' : torch.from_numpy(data['pair_type']),
            'dist_rbf'  : torch.from_numpy(data['dist_rbf']),
            'dist_bins' : torch.from_numpy(data['dist_bins']),
            'orient'    : torch.from_numpy(data['orient']),
            'dihed'     : torch.from_numpy(data['dihed']),
            'valid_mask': torch.from_numpy(data['valid_mask']),
            'rel_pos'   : torch.from_numpy(data['rel_pos']),
            # Template features (novel)
            'tmpl_dgram'    : torch.from_numpy(data['tmpl_dgram']),
            'tmpl_frames'   : torch.from_numpy(data['tmpl_frames']),
            'tmpl_torsion'  : torch.from_numpy(data['tmpl_torsion']),
            'tmpl_laplacian': torch.from_numpy(data['tmpl_laplacian']),
            'tmpl_seq_match': torch.from_numpy(data['tmpl_seq_match']),
            'tmpl_valid'    : torch.from_numpy(data['tmpl_valid']),
            'tmpl_weights'  : torch.from_numpy(data['tmpl_weights']),
        }


def collate_fn(batch: List[dict]) -> dict:
    """Pad batch to max length along L dimension."""
    keys_1d  = ['ss_pair', 'f1', 'dihed', 'valid_mask',
                 'tmpl_laplacian', 'tmpl_seq_match', 'tmpl_valid', 'tmpl_torsion']
    keys_2d  = ['MIp', 'FNp', 'contact_ss', 'rel_pos']
    keys_3d  = ['dist_rbf', 'dist_bins', 'orient', 'pair_type']
    keys_tmpl2d = ['tmpl_dgram']
    keys_frames  = ['tmpl_frames']

    max_len = max(s['seq_len'] for s in batch)
    B       = len(batch)
    out     = {'target_id': [s['target_id'] for s in batch],
               'seq_len'  : torch.tensor([s['seq_len'] for s in batch])}

    # seq_ids and seq_mask need to be built from target_id
    # (done in trainer using stored sequence)
    out['seq_mask'] = torch.zeros(B, max_len, dtype=torch.bool)
    for i, s in enumerate(batch):
        out['seq_mask'][i, :s['seq_len']] = True

    # 1-D features: (L,) or (L, d)
    for k in keys_1d:
        sample = batch[0][k]
        if sample.dim() == 1:
            out[k] = torch.zeros(B, max_len, dtype=sample.dtype)
            for i, s in enumerate(batch):
                L = s['seq_len']
                out[k][i, :L] = s[k][:L]
        else:
            d = sample.shape[-1]
            out[k] = torch.zeros(B, max_len, d, dtype=sample.dtype)
            for i, s in enumerate(batch):
                L = s['seq_len']
                out[k][i, :L] = s[k][:L]

    # 2-D features: (L, L) or (L, L, d)
    for k in keys_2d + keys_3d:
        sample = batch[0][k]
        if sample.dim() == 2:
            out[k] = torch.zeros(B, max_len, max_len, dtype=sample.dtype)
            for i, s in enumerate(batch):
                L = s['seq_len']
                out[k][i, :L, :L] = s[k][:L, :L]
        else:
            d = sample.shape[-1]
            out[k] = torch.zeros(B, max_len, max_len, d, dtype=sample.dtype)
            for i, s in enumerate(batch):
                L = s['seq_len']
                out[k][i, :L, :L] = s[k][:L, :L]

    # Template distogram: (T, L, L, 38)
    for k in keys_tmpl2d:
        sample = batch[0][k]
        T_  = sample.shape[0]
        d   = sample.shape[-1]
        out[k] = torch.zeros(B, T_, max_len, max_len, d, dtype=sample.dtype)
        for i, s in enumerate(batch):
            L = s['seq_len']
            out[k][i, :, :L, :L] = s[k][:, :L, :L]

    # Template frames: (T, L, 4, 4)
    for k in keys_frames:
        sample = batch[0][k]
        T_  = sample.shape[0]
        ident = torch.eye(4).view(1, 1, 4, 4)
        out[k] = ident.expand(B, T_, max_len, 4, 4).clone()
        for i, s in enumerate(batch):
            L = s['seq_len']
            out[k][i, :, :L] = s[k][:, :L]

    # Scalar template features: (T,)
    out['tmpl_weights'] = torch.stack([s['tmpl_weights'] for s in batch], dim=0)

    return out
