import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleInteraction(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w = nn.Linear(config["enc_dim"], 1)

    def forward(self, H):
        scores = self.w(H)
        weights = torch.softmax(scores, dim=1)
        vec = torch.sum(weights * H, dim=1)
        return vec


class MutalInteraction(nn.Module):
    def __init__(self, config):
        super().__init__()

        label_dim = config["label_dim"]
        enc_dim = config["enc_dim"]

        self.w_t = nn.Linear(enc_dim, 1)
        self.w_l = nn.Linear(label_dim, 1)

    def forward(self, H, E):
        H_t = F.gelu(self.w_t(H))
        E_t = F.gelu(self.w_l(E))

        beta = torch.einsum('bnr,lr->bnl', H_t, E_t)
        beta = beta - beta.max(dim=1, keepdim=True).values
        attn = torch.softmax(beta, dim=1)
        H_tilde = torch.einsum('bnl,bnd->bld', attn, H)
        h_tilde, _ = torch.max(H_tilde, dim=1)

        h_avg = torch.mean(H, dim=1)
        beta_d2l = F.gelu(self.w_t(h_avg)).unsqueeze(1) * F.gelu(self.w_l(E))
        h_bar = torch.mean(beta_d2l * E.unsqueeze(0), dim=1)

        h_f = torch.cat([h_tilde, h_bar], dim=1)
        return h_f
