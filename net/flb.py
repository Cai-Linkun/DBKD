import os
import torch
import torch.nn as nn
import numpy as np

from .doc_encoder import DocEncoder
from .man import MutalInteraction
from .classifier import Classifier
from .label_encoder import LabelEncoder


class FeatureBranch(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.doc = DocEncoder(config)
        self.le = LabelEncoder(config)
        self.man = MutalInteraction(config)

        c2idx = np.load(config["c2ind"], allow_pickle=True).item()
        self._hidden_dim = config["enc_dim"] + config["label_dim"]
        self.cls = Classifier(self._hidden_dim, len(c2idx), config["droprate"])

        self.persistent_path = config["persistent"]["feat"]
    
    def hidden_dim(self) -> int:
        return self._hidden_dim
    
    def forward(self, x):
        H = self.doc(x)
        E = self.le(x)
        h_f = self.man(H, E)
        p = self.cls(h_f)
        return (h_f, p)

    def load(self):
        if not os.path.exists(self.persistent_path):
            assert ("not found module params at ", self.persistent_path)

        self.load_state_dict(torch.load(self.persistent_path))
        for param in self.parameters():
            param.requires_grad = False

    def save(self):
        torch.save(self.state_dict(), self.persistent_path)
