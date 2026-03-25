"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  RNA 3D FOLDING — TEMPLATE-GUIDED TRAINING PIPELINE                          ║
║                                                                               ║
║  KEY CHANGES vs v3:                                                           ║
║   • Template database built from train PDB_RNA structures                    ║
║   • Template features cached alongside sequence features                     ║
║   • RNAFoldTemplate model with 5 novel components                            ║
║   • 9-component loss with pLDDT distillation, torsion, template FAPE        ║
║   • Sequence IDs built at collate time from stored CSV sequences             ║
║   • pLDDT-based filtering in submission (low-confidence residues flagged)    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os, gc, math, random, warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from Bio.PDB import MMCIFParser

from rna_features_v2 import find_msa_file, chunk_sequence, stitch_coords
from rna_model_template import RNAFoldTemplate, build_model_dual_gpu, cfg as ModelCfg
from rna_template_cache import (
    TemplateRNADataset, collate_fn,
    precompute_split, gpu_distance_features,
)
from rna_losses_template import compute_total_loss

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# TRAINING CONFIG
# ─────────────────────────────────────────────────────────────
class TrainConfig:
    BASE      = "/kaggle/input/competitions/stanford-rna-3d-folding-2"
    MSA_DIR   = f"{BASE}/MSA"
    PDB_DIR   = f"{BASE}/PDB_RNA"
    TRAIN_CSV = f"{BASE}/train_sequences.csv"
    VALID_CSV = f"{BASE}/validation_sequences.csv"
    TEST_CSV  = f"{BASE}/test_sequences.csv"
    OUT_DIR   = "/kaggle/working"
    CACHE_DIR = "/kaggle/working/feat_cache_tmpl"

    TRAIN_FRAC   = 1.0
    MAX_LEN      = ModelCfg.MAX_LEN
    MAX_TEMPLATES = ModelCfg.MAX_TEMPLATES

    BATCH_SIZE   = 2          # templates increase memory significantly
    EPOCHS       = 50
    LR           = 8e-5
    WEIGHT_DECAY = 0.01
    GRAD_CLIP    = 1.0
    WARMUP_STEPS = 2000
    MIXED_PREC   = True

    # ── Loss Weights ─────────────────────────────────────
    W_COORD     = 1.0
    W_TM        = 0.5
    W_FAPE      = 0.5
    W_LDDT      = 0.4    # novel
    W_PLDDT     = 0.3    # novel
    W_TORSION   = 0.3    # novel
    W_TMPL_FAPE = 0.2    # novel
    W_DIST      = 0.2
    W_RECYCLE   = 0.2

    NUM_WORKERS  = 4
    PIN_MEMORY   = True
    SEED         = 42


cfg = TrainConfig()

RNA_MAP = {
    'A':'A','U':'U','G':'G','C':'C',
    'RA':'A','RU':'U','RG':'G','RC':'C',
    '2MG':'G','1MG':'G','7MG':'G','5MC':'C',
    'H2U':'U','PSU':'U','I':'G','M2G':'G',
}
VOCAB = {'A':0,'U':1,'G':2,'C':3,'<PAD>':4,'<UNK>':5}
_cif_parser = MMCIFParser(QUIET=True)


def seed_all(seed=cfg.SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

seed_all()


# ─────────────────────────────────────────────────────────────
# CIF LOADING (from v3, unchanged)
# ─────────────────────────────────────────────────────────────
def find_cif_file(target_id: str) -> Optional[Path]:
    pdb_dir = Path(cfg.PDB_DIR)
    for name in [f"{target_id.lower()}.cif", f"{target_id.upper()}.cif", f"{target_id}.cif"]:
        p = pdb_dir / name
        if p.exists():
            return p
    return None


def load_cif_coords(target_id: str) -> Optional[Tuple[str, np.ndarray]]:
    p = find_cif_file(target_id)
    if p is None:
        return None
    try:
        structure = _cif_parser.get_structure(target_id, str(p))
        model     = structure[0]
        seq, coords = [], []
        for chain in model:
            for res in chain:
                rn   = res.resname.strip()
                base = RNA_MAP.get(rn, RNA_MAP.get(rn[-1] if rn else 'X', None))
                if base is None: continue
                seq.append(base)
                coords.append(res["C1'"].coord if "C1'" in res else [np.nan]*3)
        if not seq: return None
        return ''.join(seq), np.array(coords, np.float32)
    except Exception as e:
        print(f"  [CIF warn] {target_id}: {e}"); return None


def align_coords(csv_seq, cif_seq, cif_coords, max_len):
    L   = min(len(csv_seq), max_len)
    seq = csv_seq[:L]
    idx = cif_seq.find(seq)
    c   = cif_coords[idx:idx+L] if idx >= 0 else cif_coords[:L]
    if len(c) < L:
        pad = np.full((L - len(c), 3), np.nan, np.float32)
        c   = np.concatenate([c, pad])
    return c


def tokenise_seq(seq: str, max_len: int) -> np.ndarray:
    ids = [VOCAB.get(c, VOCAB['<UNK>']) for c in seq[:max_len]]
    ids += [VOCAB['<PAD>']] * (max_len - len(ids))
    return np.array(ids, np.int64)


# ─────────────────────────────────────────────────────────────
# TEMPLATE DATABASE BUILDER
# ─────────────────────────────────────────────────────────────
def build_template_database(df: pd.DataFrame,
                             exclude_ids: Optional[List[str]] = None
                             ) -> Dict[str, List[dict]]:
    """
    Builds a template database from training PDB_RNA structures.
    For each query, all other structures in PDB_RNA are potential templates.

    In practice on Kaggle: the full PDB_RNA directory contains known structures;
    for test sequences these are all valid templates.
    For train sequences we exclude the structure itself (leave-one-out).
    """
    print("  Building template database from PDB_RNA …")
    exclude_set = set(exclude_ids or [])

    # Load all available CIF structures
    all_structures: Dict[str, dict] = {}
    pdb_dir = Path(cfg.PDB_DIR)
    if not pdb_dir.exists():
        print("  [warn] PDB_RNA not found — templates will be empty")
        return {}

    for cif_path in tqdm(list(pdb_dir.glob("*.cif"))[:2000],
                         desc="  Loading CIF structures"):
        tid    = cif_path.stem.upper()
        result = load_cif_coords(tid)
        if result is None: continue
        cif_seq, cif_coords = result
        all_structures[tid] = {'id': tid, 'seq': cif_seq, 'coords': cif_coords}

    print(f"  Loaded {len(all_structures)} template structures")

    # For each query, build its candidate list (excluding self for train)
    template_db: Dict[str, List[dict]] = {}
    for _, row in df.iterrows():
        tid = row['target_id'].upper()
        candidates = [v for k, v in all_structures.items()
                      if k != tid and k not in exclude_set]
        # Pre-sort by rough sequence similarity (speed up downstream search)
        query_seq = str(row['sequence'])
        def rough_score(t):
            mn = min(len(query_seq), len(t['seq']))
            matches = sum(a == b for a, b in zip(query_seq[:mn], t['seq'][:mn]))
            return matches / max(len(query_seq), len(t['seq']))
        candidates.sort(key=rough_score, reverse=True)
        template_db[tid] = candidates[:50]  # keep top-50 candidates for search

    return template_db


# ─────────────────────────────────────────────────────────────
# ROW DICT BUILDER
# ─────────────────────────────────────────────────────────────
def _build_row_dicts(df: pd.DataFrame, split: str,
                     seq_dict: Dict[str, str]) -> List[dict]:
    rows = []
    for _, row in df.iterrows():
        tid = row['target_id']
        seq = str(row['sequence'])
        seq_dict[tid] = seq   # store for tokenisation at collate time

        coords = None
        if split != 'test':
            result = load_cif_coords(tid)
            if result is not None:
                cif_seq, cif_coords = result
                coords = align_coords(seq, cif_seq, cif_coords, cfg.MAX_LEN)
        rows.append({'target_id': tid, 'sequence': seq, 'split': split, 'coords': coords})
    return rows


# ─────────────────────────────────────────────────────────────
# SEQUENCE DICT (shared state for collate)
# ─────────────────────────────────────────────────────────────
_seq_dict: Dict[str, str] = {}


class SeqInjectDataset(TemplateRNADataset):
    """Adds seq_ids and true_coords from memory after cache load."""
    def __init__(self, cache_dir, split, ids, seq_dict, coords_dict):
        super().__init__(cache_dir, split, ids)
        self.seq_dict    = seq_dict
        self.coords_dict = coords_dict

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        tid  = item['target_id']
        seq  = self.seq_dict.get(tid, 'A')
        L    = item['seq_len']

        item['seq_ids']    = torch.tensor(tokenise_seq(seq, cfg.MAX_LEN), dtype=torch.long)
        true_c             = self.coords_dict.get(tid)
        if true_c is not None:
            tc = torch.from_numpy(true_c[:L])
            padded = torch.zeros(cfg.MAX_LEN, 3)
            padded[:L] = tc
            item['true_coords']  = padded
            item['true_torsion'] = torch.zeros(cfg.MAX_LEN, 6)  # filled by cache
        else:
            item['true_coords']  = torch.zeros(cfg.MAX_LEN, 3)
            item['true_torsion'] = torch.zeros(cfg.MAX_LEN, 6)
        return item


def seq_inject_collate(batch):
    base = collate_fn(batch)
    B, L = len(batch), base['seq_mask'].shape[1]

    # seq_ids
    base['seq_ids'] = torch.stack([s['seq_ids'][:L] for s in batch], 0)
    # true_coords / true_torsion
    base['true_coords']  = torch.stack([s['true_coords'][:L]  for s in batch], 0)
    base['true_torsion'] = torch.stack([s['true_torsion'][:L] for s in batch], 0)
    return base


# ─────────────────────────────────────────────────────────────
# LEARNING RATE SCHEDULE
# ─────────────────────────────────────────────────────────────
def warmup_cosine_lr(step: int, warmup: int, total: int, min_lr_frac: float = 0.1) -> float:
    if step < warmup:
        return float(step) / float(max(1, warmup))
    progress = float(step - warmup) / float(max(1, total - warmup))
    return min_lr_frac + (1 - min_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress))


# ─────────────────────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────
def run_training(train_frac  = cfg.TRAIN_FRAC,
                 epochs      = cfg.EPOCHS,
                 batch_size  = cfg.BATCH_SIZE):
    seed_all()
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _ = build_model_dual_gpu()
    base_model = model.module if hasattr(model, 'module') else model

    # ── Load CSVs ─────────────────────────────────────────
    train_df = pd.read_csv(cfg.TRAIN_CSV)
    valid_df = pd.read_csv(cfg.VALID_CSV)

    if train_frac < 1.0:
        n = max(1, int(len(train_df) * train_frac))
        train_df = train_df.sample(n, random_state=cfg.SEED).reset_index(drop=True)

    # ── Build template database ───────────────────────────
    tmpl_db = build_template_database(
        pd.concat([train_df, valid_df], ignore_index=True),
        exclude_ids=None,
    )

    # ── Build row dicts + seq/coord stores ────────────────
    seq_dict: Dict[str, str]       = {}
    coords_dict: Dict[str, np.ndarray] = {}

    train_rows = _build_row_dicts(train_df, 'train', seq_dict)
    valid_rows = _build_row_dicts(valid_df, 'val',   seq_dict)
    for rd in train_rows + valid_rows:
        if rd['coords'] is not None:
            coords_dict[rd['target_id']] = rd['coords']

    # ── Pre-cache features (once) ─────────────────────────
    for rows, split_name in [(train_rows, 'Train'), (valid_rows, 'Val')]:
        precompute_split(
            row_dicts    = rows,
            cache_dir    = cfg.CACHE_DIR,
            max_len      = cfg.MAX_LEN,
            msa_dir      = cfg.MSA_DIR,
            template_db  = {rd['target_id']: tmpl_db.get(rd['target_id'].upper(), [])
                            for rd in rows},
            device       = device,
            desc         = split_name,
        )

    # ── Datasets ──────────────────────────────────────────
    train_ids = [rd['target_id'] for rd in train_rows]
    valid_ids = [rd['target_id'] for rd in valid_rows]

    train_ds  = SeqInjectDataset(cfg.CACHE_DIR, 'train', train_ids, seq_dict, coords_dict)
    valid_ds  = SeqInjectDataset(cfg.CACHE_DIR, 'val',   valid_ids, seq_dict, coords_dict)

    loader_kwargs = dict(batch_size=batch_size, collate_fn=seq_inject_collate,
                         num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY,
                         persistent_workers=(cfg.NUM_WORKERS > 0),
                         prefetch_factor=2 if cfg.NUM_WORKERS > 0 else None)
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    valid_loader = DataLoader(valid_ds, shuffle=False, **loader_kwargs)

    # ── Optimiser ─────────────────────────────────────────
    param_groups = [
        {'params': [p for n, p in base_model.named_parameters()
                    if 'pair' in n or 'template' in n],
         'lr': cfg.LR * 3},
        {'params': [p for n, p in base_model.named_parameters()
                    if 'pair' not in n and 'template' not in n],
         'lr': cfg.LR},
    ]
    optimizer = AdamW(param_groups, weight_decay=cfg.WEIGHT_DECAY)
    total_steps = epochs * len(train_loader)
    scheduler   = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: warmup_cosine_lr(s, cfg.WARMUP_STEPS, total_steps))
    scaler = GradScaler(enabled=cfg.MIXED_PREC)

    ckpt_path = os.path.join(cfg.OUT_DIR, 'best_rna_template.pt')
    best_val  = float('inf')
    history   = []

    # ── Training loop ─────────────────────────────────────
    for epoch in range(1, epochs + 1):
        model.train()
        total_train, n_steps = 0.0, 0
        loss_parts: Dict[str, float] = {}

        for batch in tqdm(train_loader, desc=f"  Ep {epoch}/{epochs} [train]", leave=False):
            optimizer.zero_grad()
            with autocast(enabled=cfg.MIXED_PREC):
                outputs = model(batch, device=device)
                loss, parts = compute_total_loss(outputs, batch, cfg, device)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_train += loss.item()
            n_steps     += 1
            for k, v in parts.items():
                loss_parts[k] = loss_parts.get(k, 0.0) + (v.item() if torch.is_tensor(v) else v)

        avg_train = total_train / max(n_steps, 1)
        avg_parts = {k: v / max(n_steps, 1) for k, v in loss_parts.items()}

        # ── Validation ────────────────────────────────────
        model.eval()
        total_val, n_val = 0.0, 0
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"  Ep {epoch}/{epochs} [val]", leave=False):
                with autocast(enabled=cfg.MIXED_PREC):
                    outputs = model(batch, device=device)
                    loss, _ = compute_total_loss(outputs, batch, cfg, device)
                total_val += loss.item()
                n_val += 1

        avg_val = total_val / max(n_val, 1)
        lrs = scheduler.get_last_lr()

        print(
            f"\n  ┌─ Epoch {epoch:3d}/{epochs} ───────────────────────────────────\n"
            f"  │  Train: {avg_train:.5f}   Val: {avg_val:.5f}\n"
            f"  │  focal={avg_parts.get('focal',0):.4f}  "
            f"fape={avg_parts.get('fape',0):.4f}  "
            f"lddt={avg_parts.get('lddt',0):.4f}  "
            f"plddt={avg_parts.get('plddt',0):.4f}\n"
            f"  │  torsion={avg_parts.get('torsion',0):.4f}  "
            f"tmpl_fape={avg_parts.get('tmpl_fape',0):.4f}  "
            f"tm={avg_parts.get('tm',0):.4f}\n"
            f"  │  LR: {lrs[-1]:.2e}"
            f"\n  └{'─'*58}"
        )

        history.append({'epoch': epoch, 'train': avg_train, 'val': avg_val, **avg_parts})

        if avg_val < best_val:
            best_val = avg_val
            torch.save({'epoch': epoch, 'model': base_model.state_dict(),
                        'val_loss': best_val, 'history': history}, ckpt_path)
            print(f"     ✅  Saved best model (val={best_val:.4f})")

        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    pd.DataFrame(history).to_csv(
        os.path.join(cfg.OUT_DIR, 'training_history_template.csv'), index=False)
    print(f"\n  Training complete. Best val loss: {best_val:.4f}")
    return model, history, device


# ─────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model=None, device=None):
    if model is None:
        ckpt_path = os.path.join(cfg.OUT_DIR, 'best_rna_template.pt')
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"No checkpoint at {ckpt_path}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        base   = RNAFoldTemplate().to(device)
        state  = torch.load(ckpt_path, map_location=device)
        base.load_state_dict(state['model'])
        model  = base
        if torch.cuda.device_count() >= 2:
            model = nn.DataParallel(model)
        model  = model.to(device)
        print(f"  Loaded checkpoint epoch={state['epoch']}  val={state['val_loss']:.4f}")

    if device is None:
        device = next(model.parameters()).device

    for csv_path, is_test, out_name in [
        (cfg.VALID_CSV, False, 'predictions_validation.csv'),
        (cfg.TEST_CSV,  True,  'predictions_test.csv'),
    ]:
        if not Path(csv_path).exists():
            print(f"  [skip] {csv_path}"); continue

        df       = pd.read_csv(csv_path)
        split    = 'test' if is_test else 'val'
        seq_dict: Dict[str, str] = {}
        rows = _build_row_dicts(df, split, seq_dict)

        # Build empty template DB (test sequences have no coords for self)
        all_df   = pd.read_csv(cfg.TRAIN_CSV)
        tmpl_db  = build_template_database(all_df, exclude_ids=None)

        precompute_split(rows, cfg.CACHE_DIR, cfg.MAX_LEN, cfg.MSA_DIR,
                         template_db={rd['target_id']: tmpl_db.get(rd['target_id'].upper(), [])
                                      for rd in rows},
                         device=device, desc=split.capitalize())

        ids = [rd['target_id'] for rd in rows]
        ds  = SeqInjectDataset(cfg.CACHE_DIR, split, ids, seq_dict, {})
        loader = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                            collate_fn=seq_inject_collate,
                            num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY)

        model.eval()
        rows_out = []
        for batch in tqdm(loader, desc=f"  Predict {split}"):
            with autocast(enabled=cfg.MIXED_PREC):
                outputs = model(batch, device=device)
            coords = outputs['coords'].cpu().float().numpy()
            plddt  = outputs['plddt'].cpu().float().numpy()
            for b, tid in enumerate(batch['target_id']):
                L = batch['seq_len'][b]
                for i in range(L):
                    rows_out.append({
                        'target_id': tid, 'resid': i + 1,
                        'x_1': float(coords[b, i, 0]),
                        'y_1': float(coords[b, i, 1]),
                        'z_1': float(coords[b, i, 2]),
                        'plddt': float(plddt[b, i]),
                    })

        out_df = pd.DataFrame(rows_out)
        out_path = os.path.join(cfg.OUT_DIR, out_name)
        out_df.to_csv(out_path, index=False)
        low_conf = (out_df['plddt'] < 50).mean() * 100
        print(f"  Saved → {out_path}  ({len(out_df):,} rows, "
              f"{low_conf:.1f}% low-confidence residues)")


# ─────────────────────────────────────────────────────────────
# EVALUATION (with pLDDT report)
# ─────────────────────────────────────────────────────────────
def evaluate_on_validation():
    pred_path = os.path.join(cfg.OUT_DIR, 'predictions_validation.csv')
    if not Path(pred_path).exists():
        print("No validation predictions."); return

    pred_df  = pd.read_csv(pred_path)
    valid_df = pd.read_csv(cfg.VALID_CSV)
    results  = []

    for _, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc="Evaluating"):
        tid    = row['target_id']
        seq    = str(row['sequence'])
        result = load_cif_coords(tid)
        if result is None: continue
        cif_seq, cif_coords = result
        aligned = align_coords(seq, cif_seq, cif_coords, cfg.MAX_LEN)
        valid   = ~np.isnan(aligned[:, 0])
        if valid.sum() < 3: continue

        pg = pred_df[pred_df['target_id'] == tid].sort_values('resid')
        if len(pg) == 0: continue

        true_c = aligned[valid]
        pred_c = pg[['x_1','y_1','z_1']].values.astype(np.float32)
        plddt  = pg['plddt'].values if 'plddt' in pg.columns else np.ones(len(pg)) * 70

        mn = min(len(true_c), len(pred_c))
        true_c, pred_c = true_c[:mn], pred_c[:mn]

        def kabsch(P, Q):
            P = P - P.mean(0); Q = Q - Q.mean(0)
            H = P.T @ Q
            U, S, Vt = np.linalg.svd(H)
            D = np.diag([1, 1, np.linalg.det(Vt.T @ U.T)])
            return P @ (Vt.T @ D @ U.T).T, Q

        P_rot, Q = kabsch(pred_c, true_c)
        rmsd = float(np.sqrt(((P_rot - Q)**2).sum(-1).mean()))
        d0   = max(1.24*(mn-15)**(1/3)-1.8, 0.5) if mn > 21 else 0.5
        tm   = float((1/(1+((P_rot-Q)**2).sum(-1)/d0**2)).mean())
        mean_plddt = float(plddt[:mn].mean())
        results.append({'target_id': tid, 'L': mn, 'RMSD': rmsd,
                         'TM': tm, 'mean_pLDDT': mean_plddt})

    df = pd.DataFrame(results)
    if len(df):
        print(f"\n  {'='*55}")
        print(f"  Template-Guided Validation ({len(df)} targets)")
        print(f"  Mean RMSD     : {df['RMSD'].mean():.3f} Å")
        print(f"  Mean TM-score : {df['TM'].mean():.4f}")
        print(f"  TM > 0.5      : {(df['TM'] > 0.5).mean()*100:.1f}%")
        print(f"  Mean pLDDT    : {df['mean_pLDDT'].mean():.1f}")
        print(f"  {'='*55}")
        df.to_csv(os.path.join(cfg.OUT_DIR, 'validation_metrics_template.csv'), index=False)
    return df


# ─────────────────────────────────────────────────────────────
# SUBMISSION
# ─────────────────────────────────────────────────────────────
def refine_and_submit(pred_path: str, out_path: str,
                      plddt_threshold: float = 30.0):
    """Build submission CSV. Residues with pLDDT < threshold are flagged."""
    try:
        from rna_physics_refinement import post_process_predictions, format_submission
        df  = pd.read_csv(pred_path)
        df  = post_process_predictions(df, apply_physics=True)
        sub = format_submission(df, out_path)
    except ImportError:
        df  = pd.read_csv(pred_path)
        df['ID'] = df['target_id'] + '_' + df['resid'].astype(str)
        sub = df[['ID', 'x_1', 'y_1', 'z_1']]
        sub.to_csv(out_path, index=False)
        print(f"  Submission saved → {out_path}")

    if 'plddt' in df.columns:
        low = (df['plddt'] < plddt_threshold).sum()
        print(f"  Low-confidence residues (<{plddt_threshold} pLDDT): {low} ({100*low/len(df):.1f}%)")
    return sub


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    mode       = sys.argv[1] if len(sys.argv) > 1 else "full"
    train_frac = float(sys.argv[2]) if len(sys.argv) > 2 else cfg.TRAIN_FRAC
    epochs     = int(sys.argv[3])   if len(sys.argv) > 3 else cfg.EPOCHS
    batch_size = int(sys.argv[4])   if len(sys.argv) > 4 else cfg.BATCH_SIZE

    model, device = None, None
    if mode in ("train", "full"):
        model, history, device = run_training(train_frac, epochs, batch_size)
    if mode in ("infer", "full"):
        run_inference(model, device)
    if mode in ("eval", "full"):
        evaluate_on_validation()
    if mode in ("submit", "full"):
        sub_path  = os.path.join(cfg.OUT_DIR, 'submission.csv')
        test_pred = os.path.join(cfg.OUT_DIR, 'predictions_test.csv')
        if Path(test_pred).exists():
            refine_and_submit(test_pred, sub_path)
    print("\n🎉 Done!")
