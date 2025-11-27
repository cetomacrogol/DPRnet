# import os
# import random
# import numpy as np
# import torch
# import dgl
# import logging
#
# CHARSMILESSET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
#                  "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
#                  "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
#                  "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
#                  "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
#                  "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
#                  "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
#                  "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}
#
# CELLS = [
#     "HL60", "K562", "MCF7", "KB - 3 - 1", "SKOV3", "A2780", "A549", "HepG2"
# ]
# DRUGS = [
#     "doxorubicin", "vincristine", "vinblastine", "fulvestrant", "cisplatin", "paclitaxel",
#     "mitoxantrone", "tamoxifen", "colchicine", "daunorubicin", "everolimus", "imatinib"
# ]
#
# CELLS_DICT = {cell: idx for idx, cell in enumerate(CELLS)}
# DRUGS_DICT = {drug: idx for idx, drug in enumerate(DRUGS)}
#
#
# def set_seed(seed=1000):
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#
#
# def graph_collate_func(x):
#     d, p, y, feature_tensors, cell_expression_tensors, resistant_drug_indices = zip(*x)
#     d = dgl.batch(d)
#     feature_tensors = torch.stack(feature_tensors)
#     cell_expression_tensors = torch.stack(cell_expression_tensors)
#     resistant_drug_indices = torch.stack(resistant_drug_indices)
#     return d, p, torch.tensor(y), feature_tensors, cell_expression_tensors, resistant_drug_indices
#
#
# def mkdir(path):
#     path = path.strip()
#     path = path.rstrip("\\")
#     is_exists = os.path.exists(path)
#     if not is_exists:
#         os.makedirs(path)
#
#
# def encode_smiles(smiles, max_length=144):
#     encoding = np.zeros(max_length, dtype=int)
#     for idx, char in enumerate(smiles[:max_length]):
#         if char in CHARSMILESSET:
#             encoding[idx] = CHARSMILESSET[char]
#         else:
#             logging.warning(
#                 f"Character '{char}' does not exist in SMILES category encoding, skip and treat as padding.")
#     return encoding


import os
import random
import numpy as np
import torch
import dgl
import logging

CHARSMILESSET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CELLS = [
    "HL60", "K562", "MCF7", "SKOV3", "A2780", "A549"
]
# "KB - 3 - 1"
CELLS_DICT = {cell: idx for idx, cell in enumerate(CELLS)}


def set_seed(seed=1000):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def graph_collate_func(x):
    compound_graphs, compound_smiles, drug_graphs, drug_smiles, y, cell_expression_tensors = zip(*x)
    compound_graphs = dgl.batch(compound_graphs)
    drug_graphs = dgl.batch(drug_graphs)
    cell_expression_tensors = torch.stack(cell_expression_tensors)
    return compound_graphs, compound_smiles, drug_graphs, drug_smiles, torch.tensor(y), cell_expression_tensors


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)


def encode_smiles(smiles, max_length=144):
    encoding = np.zeros(max_length, dtype=int)
    for idx, char in enumerate(smiles[:max_length]):
        if char in CHARSMILESSET:
            encoding[idx] = CHARSMILESSET[char]
        else:
            logging.warning(
                f"Character '{char}' does not exist in SMILES category encoding, skip and treat as padding.")
    return encoding