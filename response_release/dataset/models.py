"""模型定义：全部从随机初始化训练，无任何预训练权重（符合课程"无预训练模型"路径）。

- SimpleNN：宽 MLP（默认 2048-2048），weight_norm + BatchNorm + Dropout + LeakyReLU。
  结构与冠军方案的 "Simple NN" 一致（引用见报告 References）。
- ResNet1D：位置保持的一维残差 CNN（多样性成员）。
  关键设计：逐级 MaxPool 下采样 + Flatten 头，保留"位置 = 特征身份"的信息；
  全局平均池化(GAP)版已验证更差（CV 0.0199 vs 0.0182），此处不再提供。

说明：BCEWithLogitsLoss 与列平均 log loss 对齐（proper scoring rule），
因此不设类别权重、不做阈值化，输出始终是概率。
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


class Block1d(nn.Module):
    """一维残差块：两层 k=3 卷积 + BN，捷径 1x1 卷积对齐通道。"""

    def __init__(self, cin: int, cout: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(cout)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Identity() if cin == cout else nn.Conv1d(cin, cout, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(F.relu(self.bn1(self.conv1(x))))
        h = self.bn2(self.conv2(h))
        return F.relu(h + self.proj(x))


class ResNet1D(nn.Module):
    """位置保持 ResNet-1D：stem -> [Block+MaxPool]x3 -> MaxPool -> Flatten -> FC 头。"""

    def __init__(self, in_dim: int, out_dim: int, channels=(32, 64, 128, 256),
                 dropout: float = 0.3, head: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(),
        )
        stages, cin = [], channels[0]
        for c in channels[1:]:
            stages += [Block1d(cin, c, dropout), nn.MaxPool1d(2)]
            cin = c
        stages.append(nn.MaxPool1d(2))
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout + 0.1),
            nn.LazyLinear(head),
            nn.BatchNorm1d(head),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.unsqueeze(1)
        h = self.stages(self.stem(h))
        return self.head(h)


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


def build_model(name: str, in_dim: int, out_dim: int, hidden: int, dropout: float,
                profile_targets=None) -> nn.Module:
    if name == "profile_mlp":
        return ProfileMLP(in_dim, out_dim, hidden, dropout, profile_targets)
    if name == "response_cnn":
        return ResponseCNN(in_dim, out_dim, hidden, dropout)
    if name == "mlp":
        return SimpleNN(in_dim, out_dim, hidden=hidden, dropout=dropout)
    if name == "mlp3":
        # 更深的变体（3 个隐藏层），作为融合多样性成员
        return SimpleNN3(in_dim, out_dim, hidden=hidden, dropout=dropout)
    if name == "resnet":
        return ResNet1D(in_dim, out_dim, dropout=dropout)
    raise ValueError(f"未知模型: {name}")


class SimpleNN3(nn.Module):
    """in -> h -> h -> h -> out 的四层 MLP 变体（多样性成员）。"""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 1024, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, hidden),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmoothBCEwLogits(nn.Module):
    """带标签平滑的 BCEWithLogitsLoss（冠军方案 smoothing=0.001）。

    标签平滑让稀有正例的目标概率略高于 0，抑制过拟合，
    同时保持 BCE 仍是 proper scoring rule 的近似（验证/测试仍用标准 BCE 指标评估）。
    """

    def __init__(self, smoothing: float = 0.001, pos_weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        t = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, t)
