"""Randomly initialized final networks: ResponseCNN and ProfileMLP.

SimpleNN is the backbone of ProfileMLP. See ARCHITECTURE_CN.md for exact shapes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNN(nn.Module):
    """in -> 2048 -> 2048 -> out。第一层与输出层 weight_norm。

    层序与冠军方案一致：BN -> Linear(wn) -> LeakyReLU -> BN -> Dropout ->
    Linear -> LeakyReLU -> BN -> Dropout -> Linear(wn)。第一层前不加 Dropout。
    """

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 2048, dropout: float = 0.25):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_dim)
        self.dense1 = nn.utils.weight_norm(nn.Linear(in_dim, hidden))
        self.bn2 = nn.BatchNorm1d(hidden)
        self.drop2 = nn.Dropout(dropout)
        self.dense2 = nn.Linear(hidden, hidden)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.drop3 = nn.Dropout(dropout)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.dense1(self.bn1(x)))
        x = F.leaky_relu(self.dense2(self.drop2(self.bn2(x))))
        x = self.dense3(self.drop3(self.bn3(x)))
        return x  # raw logits，交给 BCEWithLogitsLoss



class ResponseCNN(nn.Module):
    """Project a relative signature onto learned positions, then gate local responses."""

    def __init__(self, in_dim, out_dim, hidden=4096, dropout=0.2):
        super().__init__()
        channels, width = 256, 384
        if hidden % (channels * 4):
            raise ValueError("Projection width must be a multiple of 1024")
        self.channels = channels
        self.project = nn.Sequential(nn.BatchNorm1d(in_dim), nn.Dropout(0.1),
                                     nn.Linear(in_dim, hidden), nn.CELU(0.1))
        self.patterns = nn.Sequential(
            nn.BatchNorm1d(channels), nn.Dropout(0.1),
            nn.Conv1d(channels, width, 5, padding=2, bias=False),
            nn.BatchNorm1d(width), nn.SiLU(), nn.AvgPool1d(2),
            nn.Dropout(0.1), nn.Conv1d(width, width, 3, padding=1, bias=False),
            nn.BatchNorm1d(width), nn.SiLU())
        self.gate = nn.Sequential(nn.Dropout(dropout), nn.Conv1d(width, width, 3, padding=1),
                                  nn.BatchNorm1d(width), nn.SiLU(),
                                  nn.Conv1d(width, width, 1), nn.Sigmoid())
        n_flat = width * (hidden // channels // 4)
        self.head = nn.Sequential(nn.MaxPool1d(2), nn.Flatten(), nn.BatchNorm1d(n_flat),
                                  nn.Dropout(dropout), nn.Linear(n_flat, out_dim))

    def forward(self, x):
        positions = self.project(x).reshape(len(x), self.channels, -1)
        patterns = self.patterns(positions)
        return self.head(patterns * (2 * self.gate(patterns)))



class ProfileMLP(SimpleNN):
    """Use the group's MLP to predict both individual labels and an annotation profile."""

    def __init__(self, in_dim, out_dim, hidden, dropout, profile_targets):
        profiles = torch.as_tensor(profile_targets, dtype=torch.uint8)
        if profiles.ndim != 2 or profiles.shape[1] != out_dim or not len(profiles):
            raise ValueError("Expected fold-local profiles by scored target")
        super().__init__(in_dim, out_dim + len(profiles), hidden, dropout)
        self.register_buffer("profile_targets", profiles)
        self.n_scored = out_dim

    def logits(self, x):
        outputs = super().forward(x).float()
        return outputs[:, :self.n_scored], outputs[:, self.n_scored:]

    def training_loss(self, xb, yb, criterion):
        direct, profile = self.logits(xb)
        # One profile log likelihood alongside the sum of 206 binary log likelihoods.
        return criterion(direct, yb[:, :-1]) + F.cross_entropy(profile, yb[:, -1].long()) / self.n_scored

    def forward(self, x):
        direct, profile = self.logits(x)
        marginal = torch.softmax(profile, dim=1) @ self.profile_targets.float()
        probability = 0.5 * (torch.sigmoid(direct) + marginal)
        return torch.logit(probability.clamp(1e-15, 1 - torch.finfo(probability.dtype).eps))



def build_model(name, in_dim, out_dim, hidden, dropout, profile_targets=None):
    if name == "profile_mlp":
        return ProfileMLP(in_dim, out_dim, hidden, dropout, profile_targets)
    if name == "response_cnn":
        return ResponseCNN(in_dim, out_dim, hidden, dropout)
    if name == "mlp":
        return SimpleNN(in_dim, out_dim, hidden, dropout)
    raise ValueError(f"Unknown model: {name}")


class SmoothBCEwLogits(nn.Module):
    """带标签平滑的 BCEWithLogitsLoss（冠军方案 smoothing=0.001）。

    标签平滑把负例目标从 0 改为 0.0005、正例从 1 改为 0.9995，
    同时保持 BCE 仍是 proper scoring rule 的近似（验证/测试仍用标准 BCE 指标评估）。
    """

    def __init__(self, smoothing: float = 0.001, pos_weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        t = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, t)

