import torch
import torch.nn.functional as F
import numpy as np
from torch import nn

from .clb import ClassifierBranch
from .flb import FeatureBranch
from .stud import MainBranch
from .bhl import BHLLoss


def logits_distillation_loss(student_logits, teacher_logits, temperature: float):
    T = float(temperature)
    with torch.no_grad():
        t_prob = torch.sigmoid(teacher_logits / T) 
    kd = F.binary_cross_entropy_with_logits(
        student_logits / T, t_prob, reduction="mean" 
    ) * (T * T)
    return kd

def feat_distillation_loss(student_feat, teacher_feat):
    s = F.normalize(student_feat, p=2, dim=1)
    t = F.normalize(teacher_feat, p=2, dim=1)
    cos = (s * t).sum(dim=1)
    return (1.0 - cos).mean()

def load_pos_weight(path, device, num_labels: int | None = None) -> torch.Tensor:
    w = np.load(path, allow_pickle=True)
    if w.ndim != 1:
        w = w.reshape(-1)
    w = w.astype(np.float32, copy=False)
    return torch.from_numpy(w).to(device)


class DBKBFramework(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        device = config["device"]
        self._join_train: bool = config["join_train"]
        self.temperature = config["temperature"]

        num_labels = len(np.load(config["c2ind"], allow_pickle=True).item())
        pos_weight = load_pos_weight(config["pos_weight"], device=device, num_labels=num_labels)
        self.register_buffer("pos_weight", pos_weight)

        self._mlb = MainBranch(config)

        self._clb = ClassifierBranch(config)
        self.bhl = BHLLoss(config)

        self._flb = FeatureBranch(config)

        s_dim = self._mlb.hidden_dim()
        f_dim = self._flb.hidden_dim()

        self.s2f = nn.Sequential(
            nn.Linear(s_dim, f_dim, bias=True),
            nn.LayerNorm(f_dim),
        )


        if not self._join_train:
            self._clb.load()
            self._flb.load()

    def forward(self, x, y, adj):
        (h_f, logits_f) = self._flb(x)
        logits_c = self._clb(x)
        h_s, logits_s = self._mlb(x)

        ll = logits_distillation_loss(logits_s, logits_c.detach(), self.temperature)

        h_s_proj = self.s2f(h_s)
        fl = feat_distillation_loss(h_s_proj, h_f.detach())
        distillation_loss = adj * fl + (1 - adj) * ll

        bce_s = F.binary_cross_entropy_with_logits(
            logits_s, y, pos_weight=self.pos_weight, reduction="mean"
        )

        total_loss = distillation_loss + bce_s

        if self._join_train:
            bce_f = F.binary_cross_entropy_with_logits(
                logits_f, y, pos_weight=self.pos_weight, reduction="mean"
            )
            bhl = self.bhl(logits_c, y)
            total_loss += bhl + bce_f

        return logits_s, total_loss
