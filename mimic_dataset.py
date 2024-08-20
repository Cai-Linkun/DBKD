from typing import List
from torch.utils.data import Dataset, DataLoader
import numpy as np
import csv
import torch


def load_vocab_dict(embed_file):
    map = {}
    with open(embed_file, "r") as vocabfile:
        for i, line in enumerate(vocabfile):
            line = line.rstrip().split()
            map[line[0]] = i
    return map


class _Sample:
    def __init__(self, hadm_id, word_indices, labels):
        self._hadm_id = hadm_id
        self._word_indices = word_indices
        self._labels = labels

    def hadm_id(self):
        return self._hadm_id

    def word_indices(self):
        # doc_encoder -> embedding
        ts = torch.tensor(self._word_indices)
        ts.requires_grad_(False)
        return ts

    def pad_to(self, length):
        cur_len = len(self._word_indices)
        if cur_len < length:
            self._word_indices.extend([0] * (length - cur_len))

    def labels(self):
        ts = torch.Tensor(self._labels)
        ts.requires_grad_(False)
        return ts


class MimicData(Dataset):
    def __init__(self, config, stage) -> None:
        super().__init__()
        self.config = config
        self.c2idx = np.load(config["c2ind"], allow_pickle=True).item()
        self.w2idx = load_vocab_dict(config["embedding"])
        self.data_path = config["train_data"].replace("train", stage)
        self.max_length = config["max_words"]
        self.length = self.max_length

        self.samples: List[_Sample] = []
        self.load_data()
        self.pad()

    def pad(self):
        for sample in self.samples:
            sample.pad_to(self.length)

    def data_iters(self):
        with open(self.data_path, "r") as f:
            lr = csv.reader(f)
            next(lr)
            for row in lr:
                yield row

    def load_data(self):
        for sample in self.data_iters():
            self.samples.append(self.load_item_data(sample))

    def load_item_data(self, sample):
        # SUBJECT_ID,HADM_ID,TEXT,LABELS,length
        hadm_id = int(sample[1])
        text = sample[2]
        length = int(sample[4])
        labels_idx = np.zeros(len(self.c2idx))
        # multi-hot vector
        for l in sample[3].split(";"):
            if l in self.c2idx.keys():
                code = int(self.c2idx[l])
                labels_idx[code] = 1

        text = [int(self.w2idx[w]) if w in self.w2idx else 0 for w in text.split()]

        self.length = min(self.max_length, length)
        if len(text) > self.max_length:
            text = text[: self.max_length]

        return _Sample(hadm_id, text, labels_idx)

    def __getitem__(self, index):
        sample = self.samples[index]
        return (sample.word_indices(), sample.labels())

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    import yaml

    config = {}
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)
    train_data = MimicData(config=config, stage="dev")
    train_loader = DataLoader(
        dataset=train_data, batch_size=config["batch_size"], shuffle=True
    )
    for i, data in enumerate(train_loader):
        batch_id, batch_x, batch_y = data
        break
