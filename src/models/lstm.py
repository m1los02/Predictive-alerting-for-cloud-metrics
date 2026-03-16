from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset


class _AttentionPool(nn.Module):
    """
    Soft-attention pooling over the time dimension which learns a scalar importance score per timestep
    then returns the weighted sum of hidden states.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        scores  = self.query(h).squeeze(-1)        
        weights = torch.softmax(scores, dim=-1)
        return (h * weights.unsqueeze(-1)).sum(1)


class IncidentPredictor(nn.Module):
    """BiLSTM incident predictor"""

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_features = n_features

        self.input_norm = nn.LayerNorm(n_features)

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        d_out = hidden_dim * 2

        self.pool = _AttentionPool(d_out)

        self.head = nn.Sequential(
            nn.LayerNorm(d_out),
            nn.Linear(d_out, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x       = self.input_norm(x)
        h, _    = self.lstm(x)        
        pooled  = self.pool(h)       
        return self.head(pooled).squeeze(-1)


def _to_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(X.astype(np.float32)),
        torch.from_numpy(y.astype(np.float32)),
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=0,
    )


def _normalise_params(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flat  = X_train.reshape(-1, X_train.shape[-1])
    mean_ = flat.mean(0).astype(np.float32)
    std_  = (flat.std(0) + 1e-8).astype(np.float32)
    return mean_, std_


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    batch_size: int = 256,
    epochs: int = 40,
    lr: float = 1e-3,
    patience: int = 10,
    checkpoint_path: Optional[Path] = None,
    device: Optional[str] = None,
) -> Tuple[IncidentPredictor, np.ndarray, np.ndarray]:
    """
    Train the BiLSTM and return the best checkpoint
    Stopping criterion: validation AUPRC
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    mean_, std_ = _normalise_params(X_train)
    X_tr_n  = ((X_train - mean_) / std_).astype(np.float32)
    X_val_n = ((X_val   - mean_) / std_).astype(np.float32)

    train_loader = _to_loader(X_tr_n,  y_train, batch_size, shuffle=True)
    val_loader   = _to_loader(X_val_n, y_val,   batch_size, shuffle=False)

    n_features = X_train.shape[2]
    model = IncidentPredictor(n_features, hidden_dim, num_layers, dropout).to(device)

    pos_weight = torch.tensor(
        [(y_train == 0).sum() / max(1.0, (y_train == 1).sum())],
        dtype=torch.float32, device=device,
    )
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    amp_scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    ckpt_path = checkpoint_path or Path("checkpoints/lstm_best.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_auprc    = -1.0
    patience_left = patience

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                loss = criterion(model(X_b), y_b)
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            total_loss += loss.item() * len(y_b)
        total_loss /= len(y_train)

        model.eval()
        logits_all = []
        with torch.no_grad():
            for X_b, _ in val_loader:
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    logits_all.append(model(X_b.to(device)).cpu().float())
        val_scores = torch.sigmoid(torch.cat(logits_all)).numpy()
        val_auprc  = average_precision_score(y_val, val_scores)

        scheduler.step()

        if val_auprc > best_auprc:
            best_auprc    = val_auprc
            patience_left = patience
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "mean": mean_,
                    "std":  std_,
                    "epoch": epoch,
                    "val_auprc": val_auprc,
                },
                ckpt_path,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    # Reload best weights
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    return model, mean_, std_


def predict_lstm(
    model: IncidentPredictor,
    X: np.ndarray,
    mean_: np.ndarray,
    std_: np.ndarray,
    batch_size: int = 512,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Run inference and return per-window incident probability scores
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    X_n     = ((X - mean_) / std_).astype(np.float32)
    loader  = _to_loader(X_n, np.zeros(len(X), dtype=np.float32), batch_size, shuffle=False)
    model   = model.eval().to(device)
    logits_all = []
    with torch.no_grad():
        for X_b, _ in loader:
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits_all.append(model(X_b.to(device)).cpu().float())
    return torch.sigmoid(torch.cat(logits_all)).numpy()
