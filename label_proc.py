import os
from os.path import expanduser
from typing import Dict, Optional, Tuple
import requests
import shutil
import torch
import csv
import numpy as np
import itertools
from tqdm import tqdm
from collections import defaultdict, Counter
import tempfile, os
from pathlib import Path

SAVE_DIR = "./processed_data"
BIO_BERT_LINK = "https://www.dropbox.com/s/dc2ki2d4jv8isrb/biobert_weights.zip?dl=1"
MODEL_SAVE_DICT = "./trained_models/"
LABEL_EMBEDDING_PATH = os.path.join(SAVE_DIR, "label_embeddings.pth")


def process_dirty_code(ori_code: str, code_map):
    if ori_code.endswith("."):
        code = ori_code[:-1]
        if code in code_map:
            return code
        code = ori_code + "0"
        if code in code_map:
            return code
        code = "0" + ori_code + "0"
        if code in code_map:
            return code

    code = "0" + ori_code
    if code in code_map:
        return code

    code = ori_code[:-1]
    if code in code_map:
        return code

    code = ori_code.split(".")[0]
    if code in code_map:
        return code

    if ori_code.find("-") != -1:
        code = ori_code + ".9"
        if code in code_map:
            return code
        code = code + "9"
        if code in code_map:
            return code

        first_code, last_code = ori_code.split("-")[:2]
        prefix = ""
        if last_code.startswith(("E", "V")):
            prefix = last_code[0]
            last_code = last_code[1:]
        last_code = int(last_code)
        last_code = last_code - 1
        code = first_code + "-" + prefix + str(last_code) + ".9"
        if code in code_map:
            return code
        code = code + "9"
        if code in code_map:
            return code

    return ori_code


def download_and_extract(tgt_dir):
    os.makedirs(tgt_dir, exist_ok=True)
    response = requests.get(BIO_BERT_LINK, stream=True)
    fp = os.path.join(tgt_dir, "biobert_weights.zip")
    if response.status_code == 200:
        with open(fp, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        print(f"download success: {fp}")
    else:
        print(f"download error: {response.status_code}")
    shutil.unpack_archive(fp, tgt_dir)
    # os.remove(fp)


def load_desc_map(desc_word_file):
    c2desc_map = {}
    with open(desc_word_file, "r") as descfile:
        r = csv.reader(descfile, delimiter=" ")
        # header
        next(r)
        for row in r:
            code = row[0]
            desc = " ".join(row[1:])
            c2desc_map[code] = desc
    return c2desc_map


def buildc2idx(train_path):
    codes = set()
    for split in ["train", "dev", "test"]:
        with open(train_path.replace("train", split), "r") as f:
            lr = csv.reader(f)
            next(lr)
            count = set()
            for row in lr:
                for code in row[3].split(";"):
                    codes.add(code)
                    count.add(code)
            print("dataset ", split, " has ", len(count), " codes")
    codes = set([c for c in codes if c != ""])
    ind2c = defaultdict(str, {i: c for i, c in enumerate(sorted(codes))})
    c2ind = {c: i for i, c in ind2c.items()}
    print("all codes num: ", len(c2ind))
    return c2ind, ind2c


def code_desc_biobert(dicts):
    from transformers import BertConfig, BertModel, BertTokenizer

    model_dir = os.path.join(MODEL_SAVE_DICT, "biobert_v1.1_pubmed")
    assert os.path.exists(model_dir)

    model = BertModel.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    # tokenizer = BertTokenizer(vocab_file=voc_dir, do_lower_case=False)
    outputs = []

    c2desc, c2ind, ind2c = dicts["c2desc"], dicts["c2ind"], dicts["ind2c"]

    for _, code in tqdm(ind2c.items()):
        desc = ""
        if code not in c2desc:
            code = process_dirty_code(code, c2desc)
        if code in c2desc:
            desc = c2desc[code]
        else:
            print("not found code {} in c2desc".format(code))
        inputs = tokenizer(desc, return_tensors="pt", padding=False, truncation=False)
        with torch.no_grad():
            out = model(**inputs)
            embedding = out.last_hidden_state  # [1,len,vsize]
            embedding = torch.mean(embedding, dim=1)
            embedding = embedding.squeeze(0)  # [vsize]
            outputs.append(embedding)
    print("len_out: ", len(outputs), "len_dict: ", len(c2ind))
    outputs = torch.stack(outputs)
    torch.save(outputs, LABEL_EMBEDDING_PATH)
    print("process success, saved to ", LABEL_EMBEDDING_PATH)


# 1. 根据 parent map,把每个样本的父标签也附加到该样本中


def build_label_hierarchy(c2desc):
    parent_map = {}
    filled = [
        ["E000-E999", "E800-E999"],
        ["E000-E999", "E846-E848"],
        ["E970-E979", "E970-E978"],
        ["E970-E979", "E979-E979"],
        ["V01-V91", "V01-V09", "V01-V06"],
        ["V01-V91", "V01-V09", "V07-V09"],
        ["V01-V91", "V01-V86"],
        ["00-99", "00"],
        ["070-079", "72-75"],
        ["001-139", "76-84"],
        ["001-139", "00-99"],
    ]

    hidx = [[], [], [], []]

    def _dirty_code(ori_code: str):
        if ori_code in c2desc:
            return ori_code
        if ori_code.find("-") != -1:
            code = ori_code + ".9"
            if code in c2desc:
                return code
            code = code + "9"
            if code in c2desc:
                return code

            first_code, last_code = ori_code.split("-")[:2]
            prefix = ""
            if last_code.startswith(("E", "V")):
                prefix = last_code[0]
                last_code = last_code[1:]
            last_code = int(last_code)
            last_code = last_code - 1
            code = first_code + "-" + prefix + str(last_code) + ".9"
            if code in c2desc:
                return code
            code = code + "9"
            if code in c2desc:
                return code
        return ori_code

    with open("./processed_data/label_hierarchy.csv", "r") as f:
        reader = csv.reader(f)
        for row in itertools.chain(reader, filled):
            for i in range(1, len(row)):
                child = _dirty_code(row[i])
                parent = _dirty_code(row[i - 1])
                if child == parent:
                    continue

                parent_map[child] = parent
                if child in c2desc:
                    hidx[i].append(child)
                if parent in c2desc:
                    hidx[i - 1].append(parent)

    for i in range(len(hidx)):
        hidx[i] = sorted(list(set(hidx[i])))

    for code, _ in c2desc.items():
        if code not in parent_map:
            found = False
            if str(code).endswith((".99", ".9")):
                _code = code.split(".")[0]
                if _code in parent_map:
                    parent_map[code] = _code
                    found = True

            if not found and code.find("-") == -1:
                _code = code[:-1]
                if _code.endswith("."):
                    _code = _code[:-1]
                if _code in c2desc:
                    parent_map[code] = _code
                    found = True

            if str(code).startswith("E"):
                _code = code[:-1]
                if _code.endswith("."):
                    _code = _code[:-1]
                if _code in c2desc:
                    parent_map[code] = _code
                    found = True

            if str(code).startswith("V"):
                _code = code[:-1]
                if _code.endswith("."):
                    _code = _code[:-1]
                if _code in c2desc:
                    parent_map[code] = _code
                    found = True

            num = -1
            if not found:
                try:
                    num = float(code)
                except ValueError:
                    print(f"Cannot convert '{code}' to a number.")
                if num >= 0 and num <= 139:
                    parent_map[code] = "001-139"
                    found = True

            if found:
                par = parent_map[code]
                if par not in c2desc:
                    if par.find("-") != -1:
                        _par = par + ".9"
                        if _par in c2desc:
                            parent_map[code] = _par
                            found = True
                        else:
                            _par = _par + "9"
                            if _par in c2desc:
                                parent_map[code] = _par
                                found = True
            if not found:
                print(code)

    filtered_parent_map = {k: v for k, v in parent_map.items() if v in c2desc}

    # index_map = {}
    # count = 0
    # for c, idx in c2ind.items():
    #     if c in parent_map:
    #         found = False
    #         p = parent_map[c]
    #         if p in c2ind:
    #             index_map[c] = p
    #             found = True
    #         else:
    #             if p in parent_map:
    #                 pp = parent_map[p]
    #                 if pp in c2ind:
    #                     index_map[c] = pp
    #                     found = True
    #         if not found:
    #             count += 1
    #             print("not found p in c2ind", p)
    #     else:
    #         count += 1
    #         print("not found c in c2ind", c)
    # print("not found count ", count)
    # for child, parent in parent_map.items():

    # for k, v in parent_map.items():
    #     if k not in c2desc:
    #         k = k + ".99"
    #         if k not in c2desc:
    #             print("not found k: ", k)
    #     if v not in c2desc:
    #         v = v + ".99"
    #         if v not in c2desc:
    #             print("not found v: ", v)
    return filtered_parent_map, hidx


def build_c2p(dicts):
    from third_party.icd9 import icd9

    c2desc, c2ind, ind2c = dicts["c2desc"], dicts["c2ind"], dicts["ind2c"]
    tree = icd9.ICD9(os.path.join(SAVE_DIR, "codes_filled.json"))
    for code, idx in c2ind.items():
        res = tree.search(code)
        if len(res) == 0:
            _code = process_dirty_code(code, c2desc)
            res = tree.search(_code)
            if len(res) == 0:
                _code = code[:-1]
                if _code.endswith("."):
                    _code = _code[:-1]
                res = tree.search(_code)
                if len(res) == 0:
                    print("not handled code: ", code)


def rewrite_train_data_with_hierarchy(train_path, hierarchy, c2desc):
    C = DictCounter("./processed_data/label_counter.npy")
    # for split in ["train", "dev", "test"]:
    for split in ["train"]:
        file_path = train_path.replace("train", split)
        data = []
        with open(file_path, "r") as f:
            lr = csv.reader(f)
            next(lr)
            for row in lr:
                new_row = row
                row_codes = row[3].split(";")
                new_codes = row_codes
                for code in row_codes:
                    _code = code
                    # Add parent nodes recusive
                    while _code in hierarchy:
                        parent_node = hierarchy[_code]
                        new_codes.append(hierarchy[_code])
                        _code = parent_node
                        # print("add code {}'s parent node {}".format(code, parent_node))
                new_row[3] = ";".join(new_codes)
                data.append(new_row)
                for code in new_codes:
                    C.add(code)
        C.save()
        file_path = train_path.replace("train", split + "_ov")
        with open(file_path, "w") as f:
            f.write(
                ",".join(["SUBJECT_ID", "HADM_ID", "TEXT", "LABELS", "length"]) + "\n"
            )
            for row in data:
                f.write(",".join(row) + "\n")
        print("write hierarchy-processed file to ", file_path)


class DictCounter:
    def __init__(self, path):
        self.path = Path(path)
        self.counter = self._load_or_init()

    def update(self, iterable):
        self.counter.update(iterable)

    def add(self, key, count=1):
        self.counter[key] += count

    def save(self):
        arr = np.array(dict(self.counter), dtype=object)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=str(self.path.parent), delete=False
        ) as tmp:
            np.save(tmp, arr, allow_pickle=True)
            tmp_name = tmp.name
        os.replace(tmp_name, self.path)

    def _load_or_init(self):
        if self.path.exists():
            try:
                arr = np.load(self.path, allow_pickle=True).item()
                return Counter(arr)
            except Exception:
                return Counter()
        return Counter()

    def __getitem__(self, key):
        return self.counter[key]

    def __repr__(self):
        return f"DictCounter({dict(self.counter)})"

def build_hm_idx_map(c2ind, hierarchy, hier_codes):
    hierarchy_index_map = {}
    hier_level_idx = []
    for k, v in hierarchy.items():
        _k = -1
        if k in c2ind:
            _k = c2ind[k]

        _v = -1
        if v in c2ind:
            _v = c2ind[v]

        if _k != -1 and _v != -1:
            hierarchy_index_map[_k] = _v
        else:
            print("not found {}:{}, {}:{}", k, v, _k, _v)

    for level_codes in hier_codes:
        hier_level_idx.append([])
        for code in level_codes:
            if code in c2ind:
                hier_level_idx[-1].append(c2ind[code])

    print("len hierarchy_index_map: ", len(hierarchy_index_map))

    return hierarchy_index_map, hier_level_idx


def build_label_co_matrix(data_path, c2idx):
    num_labels = len(c2idx)
    matrix = np.eye(num_labels, num_labels)
    count = set()
    for split in ["train", "dev", "test"]:
        file_path = data_path.replace("train", split)
        with open(file_path, "r") as f:
            lr = csv.reader(f)
            next(lr)
            for row in lr:
                row_codes = row[3].split(";")
                for c1 in row_codes:
                    for c2 in row_codes:
                        if c1 in c2idx and c2 in c2idx:
                            matrix[c2idx[c1]][c2idx[c2]] = 1
                            matrix[c2idx[c2]][c2idx[c1]] = 1
                            count.add(c1 + c2)
    print("co-cur count: ", len(count))
    return matrix


def save_pos_weight_npy(
    train_data_path,
    id2idx_path,
    out_path: str,
    clip: Tuple[float, float] = (1.0, 6.0),  # 权重裁剪区间
    add_k: float = 1.0,
):
    counts = Counter()
    total = 0
    with open(train_data_path, "r") as f:
        lr = csv.reader(f)
        next(lr) # header
        for row in lr:
            total += 1
            for l in row[3].split(";"):
                counts[l] += 1

    c2idx = np.load(id2idx_path, allow_pickle=True).item()
    C = max(c2idx.values()) + 1
    pos = np.zeros(C, dtype=np.float64)
    for k, idx in c2idx.items():
        pos[idx] = counts.get(k, 0)

    pos = pos + float(add_k)
    eps = 1e-12
    beta = 1.0 - 1.0 / float(total)   # ≈(N-1)/N
    beta = float(np.clip(beta, 0.9, 0.99999))
    E = (1.0 - np.power(beta, pos)) / (1.0 - beta)
    w = 1.0 / np.clip(E, eps, None)
    med = np.median(w)
    if med > 0:
        w = w / med

    w = np.clip(w, clip[0], clip[1]).astype(np.float32)
    np.save(out_path, w, allow_pickle=True)

if __name__ == "__main__":
    pass
    # c2desc = load_desc_map(os.path.join(SAVE_DIR, "description_words.vocab"))

    # code_desc_biobert(dicts)
    # build_c2p(dicts)
    # label_hm, hier_codes = build_label_hierarchy(c2desc)
    # ori_data_path = os.path.join(SAVE_DIR, "train_full.csv")  # change this
    # rewrite_train_data_with_hierarchy(ori_data_path, label_hm, c2desc)

    # processed_data_path = os.path.join(SAVE_DIR, "train_ov_full.csv")
    # c2ind, ind2c = buildc2idx(processed_data_path)
    # np.save(os.path.join(SAVE_DIR, "c2ind.npy"), c2ind)
    # np.save(os.path.join(SAVE_DIR, "ind2c.npy"), ind2c)
    # dicts = {}
    # dicts["c2desc"] = c2desc
    # dicts["c2ind"] = c2ind
    # dicts["ind2c"] = ind2c
    # # code_desc_biobert(dicts)
    # hierarchy_index_map, hier_level_idx = build_hm_idx_map(c2ind, label_hm, hier_codes)
    # np.save(os.path.join(SAVE_DIR, "hmidx.npy"), hierarchy_index_map)
    # np.save(
    #     os.path.join(SAVE_DIR, "hier_level_idx.npy"),
    #     np.array(hier_level_idx, dtype=object),
    # )

    # co_matrix = build_label_co_matrix(processed_data_path, c2ind)
    # np.save(os.path.join(SAVE_DIR, "comatrix.npy"), co_matrix)

    # save_pos_weight_npy(os.path.join(SAVE_DIR, "train_ov_full.csv"), os.path.join(SAVE_DIR, "c2ind.npy"),
    #                     os.path.join(SAVE_DIR, "pos_weight.npy"))
