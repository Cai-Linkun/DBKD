import numpy as np
import torch.nn as nn

from .doc_encoder import DocEncoder
from .man import SimpleInteraction
from .classifier import Classifier


class MainBranch(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.doc = DocEncoder(config)
        self.man = SimpleInteraction(config)
        self._hidden_dim = config["enc_dim"]
        c2idx = np.load(config["c2ind"], allow_pickle=True).item()
        self.cls = Classifier(self._hidden_dim, len(c2idx), config["droprate"])

    def hidden_dim(self) -> int:
        return self._hidden_dim

    def forward(self, x):
        H = self.doc(x)
        h = self.man(H)
        p = self.cls(h)
        return h, p
