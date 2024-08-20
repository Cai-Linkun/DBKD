import torch
import torch.nn as nn


class LabelEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        _in = config["bio_bert"]["vector_size"]
        _out = config["label_dim"]
        self.transform = nn.Linear(_in, _out)
        path = config["label_enc"]["label_embeddings"]
        self.register_buffer("embeddings", torch.load(path))

    def forward(self, x):
        return self.transform(self.embeddings)
