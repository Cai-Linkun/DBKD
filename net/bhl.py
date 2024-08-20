import logging
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tempfile, os
from pathlib import Path
from collections import defaultdict, Counter


logger = logging.getLogger()

def load_pos_weight(path, device, num_labels: int | None = None) -> torch.Tensor:
    w = np.load(path, allow_pickle=True)
    if w.ndim != 1:
        w = w.reshape(-1)
    w = w.astype(np.float32, copy=False)
    return torch.from_numpy(w).to(device)


def logitsToPred(logits, threshold):
    probabilities = torch.sigmoid(logits)
    predicted_labels = (probabilities >= threshold).float()
    return predicted_labels


class BHLLoss(nn.Module):
    def __init__(self, config):
        super(BHLLoss, self).__init__()
        self.hmidx = np.load(config["hmidx"], allow_pickle=True).item()
        comatrix: np.ndarray = np.load(config["comatrix"])
        comatrix = np.nan_to_num(comatrix, nan=0.0, posinf=0.0, neginf=0.0)
        device = torch.device(config["device"])
        self.comatrix: torch.Tensor = torch.from_numpy(comatrix).float().to(device)
        self.comatrix.requires_grad_(False)
        self.threshold = config["cls_threshold"]
        self.c2idx = np.load(config["c2ind"], allow_pickle=True).item()
        self.lamda = config["lambda"]
        self.gamma = config["gamma"]
        self.beta = config["beta"]
        self.hire_margin = config["hire_margin"]

        self.hier_level_idx = np.load(config["hier_level_idx"], allow_pickle=True)
        self.num_labels = len(self.c2idx)
        self.bce_logits = nn.BCEWithLogitsLoss(reduction='none')
        with torch.no_grad():
            A = self.comatrix
            deg = A.sum(dim=1)
            d_inv_sqrt = torch.zeros_like(deg)
            mask = deg > 0
            d_inv_sqrt[mask] = deg[mask].pow(-0.5)
            # D^{-1/2} A D^{-1/2}
            normA = (d_inv_sqrt.unsqueeze(1) * A) * d_inv_sqrt.unsqueeze(0)
            L = torch.eye(A.size(0), device=device, dtype=A.dtype) - normA
            L = torch.nan_to_num(L, nan=0.0, posinf=0.0, neginf=0.0)
            self.register_buffer("L", L)

        par_indices = []
        for idx in range(self.num_labels):
            pidx = self.hmidx.get(idx, self.num_labels)
            par_indices.append(pidx if pidx < self.num_labels else idx)
        self.register_buffer("parent_idx",
            torch.tensor(par_indices, dtype=torch.long, device=device))
        
        ar = torch.arange(self.num_labels, device=device)
        self.register_buffer('is_root', (self.parent_idx == ar))

        pos_weight = load_pos_weight(config["pos_weight"], device=device, num_labels=self.num_labels)
        self.register_buffer('pos_weight', pos_weight)

        children = [[] for _ in range(self.num_labels)]
        parent_idx_cpu = self.parent_idx.detach().cpu().tolist()
        for child, par in enumerate(parent_idx_cpu):
            if child != par:
                children[par].append(child)

        max_k = max((len(v) for v in children), default=0)
        if max_k == 0:
            max_k = 1 

        child_index = torch.full((self.num_labels, max_k), -1, dtype=torch.long, device=device)
        child_mask  = torch.zeros_like(child_index, dtype=torch.bool)
        for pi, lst in enumerate(children):
            if len(lst) > 0:
                idx = torch.tensor(lst, device=device, dtype=torch.long)
                child_index[pi, :len(lst)] = idx
                child_mask [pi, :len(lst)] = True

        self.register_buffer("child_index", child_index)
        self.register_buffer("child_mask",  child_mask)

        logger.info(
            "len(hmidx): {}, shape(comatrix): {}, num_labels: {}".format(
                len(self.hmidx), self.comatrix.shape, self.num_labels
            )
        )
    
    def forward(self, logits, y):
        p = torch.sigmoid(logits)
        p_parent = p.index_select(1, self.parent_idx)
        y_parent = y.index_select(1, self.parent_idx)

        bce_elem = F.binary_cross_entropy_with_logits(
                logits, y, reduction='none', pos_weight=self.pos_weight
            )
        not_root = (~self.is_root).unsqueeze(0) 
        diff_pos = (p > self.threshold) & (p_parent < self.threshold) & not_root
        weight = 1.0 + self.gamma * diff_pos.float()

        reg = torch.einsum('bc,cd,bd->b', p, self.L, p)
        reg = torch.nan_to_num(reg, nan=0.0, posinf=1e4, neginf=0.0)
        reg = reg / float(logits.size(1))

        loss = (bce_elem * weight ).mean(dim=1) + self.downward(p, y)
        loss = (1 - self.lamda) * loss + self.lamda * reg
        return loss.mean()
    
    def downward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        B, C = p.shape
        eps = 1e-6
        y_parent = y.index_select(1, self.parent_idx)
        p_parent = p.index_select(1, self.parent_idx)

        idx = self.child_index.unsqueeze(0).expand(B, -1, -1)
        one_minus_p = 1.0 - p 
        one_minus_p_exp = one_minus_p.unsqueeze(-1).expand(B, one_minus_p.size(1), idx.size(-1))
        gathered = torch.gather(one_minus_p_exp, 1, idx.clamp_min(0)) 
        gathered = torch.where(self.child_mask.unsqueeze(0), gathered, torch.ones_like(gathered))
        log_prod = torch.log(gathered.clamp(min=eps)).sum(dim=2)
        tilde_p = 1.0 - torch.exp(log_prod)
        tilde_p = tilde_p.clamp(min=eps, max=1.0 - eps)

        gate = ((p_parent > self.threshold) | y_parent.bool()).float()

        bce = F.binary_cross_entropy(tilde_p, y_parent.float(), reduction='none')
        return (self.beta * gate * bce).mean(dim=1)
