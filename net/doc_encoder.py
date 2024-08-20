import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict


class ConvLayer(nn.Module):
    def __init__(self, layer_idx, in_channels, out_channels, kernel_size):
        super().__init__()
        self.kernel_size: int = kernel_size
        padding = 0
        if kernel_size % 2 == 1:
            padding = (kernel_size - 1) // 2

        name_prefix = "convlayer_" + str(layer_idx) + "_"
        self.model = nn.Sequential(
            OrderedDict(
                [
                    (name_prefix + "norm", nn.BatchNorm1d(in_channels)),
                    (name_prefix + "relu", nn.ReLU(inplace=True)),
                    (
                        name_prefix + "conv",
                        nn.Conv1d(
                            in_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=padding,
                            bias=False,
                        ),
                    ),
                ]
            )
        )

    def forward(self, x):
        if self.kernel_size % 2 == 0:
            # keep sequence length
            zero_padding = torch.zeros(x.size(0), x.size(1), self.kernel_size - 1).to(
                x.device
            )
            x = torch.cat([x, zero_padding], dim=2)
        x = self.model(x)

        return x


class TransLayer(nn.Module):
    def __init__(self, layer_idx, in_channels, out_channels):
        super(TransLayer, self).__init__()
        name_prefix = "translayer_" + str(layer_idx) + "_"
        self.model = nn.Sequential(
            OrderedDict(
                [
                    (
                        name_prefix + "conv",
                        nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0),
                    ),
                    (name_prefix + "norm", nn.BatchNorm1d(out_channels)),
                ]
            )
        )

    def forward(self, x):
        x = self.model(x)
        return x


class Block(nn.Module):
    def __init__(self, block_idx, in_channels, out_channels, kernel_size):
        super(Block, self).__init__()
        self.conv_layer = ConvLayer(block_idx, in_channels, out_channels, kernel_size)
        self.trans_layer = TransLayer(block_idx, out_channels * 2, out_channels)

    def forward(self, x):
        out = self.conv_layer(x)
        x = torch.cat([x, out], dim=1)
        x = self.trans_layer(x)
        return x


class DenseNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_block, kernel_size, droprate=0.5):
        super(DenseNet, self).__init__()

        self.num_features = out_channels
        self.num_block = num_block
        self.kernel_size = kernel_size

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.blocks = self.create_blocks()
        self.sequence = []
        self.drops = [nn.Dropout(droprate) for _ in range(len(self.blocks))]

    def layers(self):
        return self.sequence

    def forward(self, x):
        x = self.conv1(x)
        self.sequence = []
        for drop, block in zip(self.drops, self.blocks):
            x = block(x)
            x = drop(x)
            self.sequence.append(x)
        return x

    def create_blocks(self):
        blocks = nn.ModuleList()

        for idx in range(self.num_block):
            block = Block(idx, self.num_features, self.num_features, self.kernel_size)
            blocks.append(block)
        return blocks


def load_embeddings(embed_file):
    W = []
    with open(embed_file) as ef:
        for line in ef:
            line = line.rstrip().split()
            vec = np.array(line[1:]).astype(np.float64)
            vec = vec / float(np.linalg.norm(vec) + 1e-6)
            W.append(vec)
        vec = np.random.randn(len(W[-1]))
        vec = vec / float(np.linalg.norm(vec) + 1e-6)
        W.append(vec)
    W = np.array(W)
    return W


class DocEncoder(nn.Module):
    def __init__(self, config):
        super(DocEncoder, self).__init__()
        self.config = config
        self.layer_size = config["doc_enc"]["dense_layers"]
        self.dense_net = DenseNet(
            config["w2v_size"],
            config["enc_dim"],
            self.layer_size,
            config["doc_enc"]["kernel_size"],
            config["droprate"],
        )
        self.rewight_mlp = nn.Linear(self.layer_size, self.layer_size)
        emb = load_embeddings(config["embedding"])
        self.embedding = nn.Embedding(emb.shape[0], emb.shape[1], padding_idx=0)
        self.embedding.weight = nn.Parameter(torch.from_numpy(emb).float())

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        _ = self.dense_net(x)
        X = self.dense_net.layers()

        X = torch.stack(X)
        R = torch.sum(X, dim=3)
        R = R.permute(1, 2, 0)

        weights = self.rewight_mlp(R)
        alpha = torch.softmax(weights, dim=2).permute(0, 2, 1).unsqueeze(-1)

        X = X.permute(1, 0, 2, 3)
        H = torch.sum(alpha * X, dim=1)
        H = H.permute(0, 2, 1)
        return H