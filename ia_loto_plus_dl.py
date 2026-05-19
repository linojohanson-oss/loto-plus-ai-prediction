# ia_loto_plus_dl.py
# Deep Learning para Loto Plus (Transformer)
# Compatible con versiones viejas de PyTorch (sin verbose en scheduler)

import os
import sys
import random
from typing import List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================
# CONFIG
# =========================
INPUT_XLSX = "loto_plus_incremental.xlsx"
DEFAULT_SHEET = "Tradicional"

NUM_COLS = ["n1", "n2", "n3", "n4", "n5", "n6"]
MAX_NUM = 45

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

WINDOW = 40
ROLL_R = 20

BATCH_SIZE = 128
EPOCHS = 500

LR = 8e-4
WEIGHT_DECAY = 2e-4

TRAIN_FRAC = 0.75
VAL_FRAC = 0.15

PATIENCE = 80
MIN_DELTA = 1e-7

# Scheduler
SCHED_FACTOR = 0.5
SCHED_PATIENCE = 10
SCHED_MIN_LR = 2e-5

N_CANDIDATES = 12
TEMPERATURE = 0.90

OUT_XLSX = "prediccion_dl_loto_plus.xlsx"
CKPT_DIR = "checkpoints_dl"

# =========================
# Utils
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# =========================
# Data
# =========================
def load_sheet(path, sheet):
    df = pd.read_excel(path, sheet_name=sheet)

    if "sorteo" not in df.columns:
        raise ValueError("La hoja no tiene columna 'sorteo'")

    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    df = df[df[NUM_COLS].notna().sum(axis=1) == 6].copy()
    df = df.sort_values("sorteo").reset_index(drop=True)
    return df


def draw_to_multihot(nums):
    v = np.zeros(MAX_NUM, dtype=np.float32)
    for n in nums:
        v[int(n) - 1] = 1.0
    return v


def build_multihot(df):
    return np.stack([
        draw_to_multihot([row[c] for c in NUM_COLS])
        for _, row in df.iterrows()
    ])


def build_features(X):
    roll = np.zeros_like(X)
    for t in range(len(X)):
        a = max(0, t - ROLL_R)
        roll[t] = X[a:t].mean(axis=0) if t > 0 else 0
    return np.concatenate([X, roll], axis=1)


class LotoDataset(Dataset):
    def __init__(self, F, Y, window):
        self.F = F
        self.Y = Y
        self.window = window

    def __len__(self):
        return len(self.F) - self.window

    def __getitem__(self, i):
        t = i + self.window
        return (
            torch.tensor(self.F[t - self.window:t], dtype=torch.float32),
            torch.tensor(self.Y[t], dtype=torch.float32),
        )


def temporal_split(n):
    tr = int(n * TRAIN_FRAC)
    va = int(n * (TRAIN_FRAC + VAL_FRAC))
    return slice(0, tr), slice(tr, va), slice(va, n)

# =========================
# Model
# =========================
class TransformerModel(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, 192)
        enc = nn.TransformerEncoderLayer(
            d_model=192,
            nhead=8,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=3)
        self.out = nn.Linear(192, MAX_NUM)

    def forward(self, x):
        x = self.fc(x)
        x = self.encoder(x)
        return self.out(x[:, -1])

# =========================
# Training
# =========================
def make_pos_weight(y):
    pos = y.sum(axis=0) + 1e-6
    neg = y.shape[0] - pos
    return torch.tensor(neg / pos, device=DEVICE)


def train_model(F, Y):
    ds = LotoDataset(F, Y, WINDOW)
    tr, va, _ = temporal_split(len(ds))

    dl_tr = DataLoader(torch.utils.data.Subset(ds, range(tr.start, tr.stop)),
                       batch_size=BATCH_SIZE, shuffle=True)
    dl_va = DataLoader(torch.utils.data.Subset(ds, range(va.start, va.stop)),
                       batch_size=BATCH_SIZE)

    y_train = np.concatenate([y.numpy() for _, y in dl_tr])
    pos_weight = make_pos_weight(y_train)

    model = TransformerModel(F.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=SCHED_FACTOR,
        patience=SCHED_PATIENCE,
        min_lr=SCHED_MIN_LR,
    )

    best = float("inf")
    bad = 0

    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt = f"{CKPT_DIR}/best_transformer.pt"

    for ep in range(1, EPOCHS + 1):
        model.train()
        for x, y in dl_tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(DEVICE), y.to(DEVICE)
                val_loss += loss_fn(model(x), y).item()

        scheduler.step(val_loss)

        lr_now = opt.param_groups[0]["lr"]
        print(f"Epoch {ep:03d} | lr={lr_now:.6f} | val_loss={val_loss:.6f}", flush=True)

        if val_loss < best - MIN_DELTA:
            best = val_loss
            bad = 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= PATIENCE:
                print("Early stopping ✅", flush=True)
                break

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    return model

# =========================
# Predict + Export
# =========================
@torch.no_grad()
def predict_probs(model, F):
    x = torch.tensor(F[-WINDOW:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    return torch.sigmoid(model(x)).cpu().numpy()[0]


def generate_combos(p):
    p = p / p.sum()
    out = []
    while len(out) < N_CANDIDATES:
        idx = np.random.choice(range(MAX_NUM), 6, replace=False, p=p)
        nums = sorted(i + 1 for i in idx)
        if nums not in out:
            out.append(nums)
    return out


def export_excel(probs, combos):
    df_rank = pd.DataFrame({
        "numero": range(1, 46),
        "probabilidad": probs
    }).sort_values("probabilidad", ascending=False)

    df_top20 = df_rank.head(20)
    df_combos = pd.DataFrame({
        "jugada": range(1, len(combos) + 1),
        "numeros": ["-".join(f"{n:02d}" for n in c) for c in combos],
    })

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        df_rank.to_excel(w, sheet_name="Ranking_DL", index=False)
        df_top20.to_excel(w, sheet_name="Top20_DL", index=False)
        df_combos.to_excel(w, sheet_name="Combinaciones_DL", index=False)

    print(f"\nOK ✅ Excel generado: {OUT_XLSX}")

# =========================
# MAIN
# =========================
def main():
    set_seed()
    sheet = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SHEET

    df = load_sheet(INPUT_XLSX, sheet)
    X = build_multihot(df)
    F = build_features(X)

    print(f"Sorteos: {len(X)} | DEVICE={DEVICE}")
    model = train_model(F, X)

    probs = predict_probs(model, F)
    combos = generate_combos(probs)
    export_excel(probs, combos)


if __name__ == "__main__":
    main()
