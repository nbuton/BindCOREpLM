"""
Dataset loader for the LIP_dataset format:

    >protein_id
    MSEQVENCEHERE...
    01---001100...

Three lines per protein: header (starts with '>'), sequence, and a
same-length label string over {'0', '1', '-'}, where '-' marks residues
to ignore in the loss (e.g. unresolved/unlabeled positions).

Label alignment with tokenized input
-------------------------------------
ESM-style tokenizers add special tokens (e.g. <cls> at the start, <eos>
at the end) around the residue tokens. `LIPCollator` uses the tokenizer's
own `special_tokens_mask` to figure out exactly which token positions
correspond to real residues, and places the parsed per-residue labels
there in order (everything else -- special tokens and padding -- is
labeled -100, the standard "ignore" index). This keeps the collator
correct regardless of exactly how many special tokens a given model adds.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

LABEL_IGNORE_INDEX = -100
_LABEL_CHAR_TO_INT = {"0": 0, "1": 1, "-": LABEL_IGNORE_INDEX}


def parse_lip_file(path: str) -> List[Tuple[str, str, List[int]]]:
    """Parses one LIP_dataset-format file into (protein_id, sequence, labels) tuples."""
    with open(path, "r") as f:
        lines = [line.rstrip("\n") for line in f if line.strip() != ""]

    if len(lines) % 3 != 0:
        raise ValueError(
            f"{path}: expected 3 lines per protein (header/sequence/labels), "
            f"but found {len(lines)} non-empty lines, which isn't a multiple of 3."
        )

    samples = []
    for i in range(0, len(lines), 3):
        header, sequence, label_str = lines[i], lines[i + 1], lines[i + 2]
        if not header.startswith(">"):
            raise ValueError(f"{path}: line {i} expected a '>' header, got: {header!r}")
        if len(sequence) != len(label_str):
            raise ValueError(
                f"{path}: protein {header!r} has sequence length {len(sequence)} "
                f"but label length {len(label_str)} -- they must match."
            )
        try:
            labels = [_LABEL_CHAR_TO_INT[c] for c in label_str]
        except KeyError as e:
            raise ValueError(
                f"{path}: protein {header!r} has an unexpected label character "
                f"{e.args[0]!r} (expected one of '0', '1', '-')."
            )
        protein_id = header[1:].strip()
        samples.append((protein_id, sequence, labels))
    return samples


def compute_pos_weight(samples: Sequence[Tuple[str, str, List[int]]]) -> float:
    """pos_weight = num_negative / num_positive over non-ignored residues,
    the standard value to pass to BCEWithLogitsLoss(pos_weight=...) to
    counter class imbalance (binding residues are typically the minority).
    """
    num_pos = 0
    num_neg = 0
    for _, _, labels in samples:
        for l in labels:
            if l == 1:
                num_pos += 1
            elif l == 0:
                num_neg += 1
    if num_pos == 0:
        raise ValueError("No positive (binding) residues found -- cannot compute pos_weight.")
    return num_neg / num_pos


class LIPDataset(Dataset):
    """Wraps one or more LIP_dataset-format files (e.g. pass both train.txt
    and valid.txt to build a combined training set for a final run).
    """

    def __init__(self, paths: Sequence[str]):
        self.samples: List[Tuple[str, str, List[int]]] = []
        for path in paths:
            self.samples.extend(parse_lip_file(path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        protein_id, sequence, labels = self.samples[idx]
        return {"id": protein_id, "sequence": sequence, "labels": labels}

    def pos_weight(self) -> float:
        return compute_pos_weight(self.samples)


@dataclass
class LIPCollator:
    """Tokenizes a batch of sequences and aligns residue-level labels to
    tokenized positions, ignoring special tokens and padding.
    """

    tokenizer: object
    max_length: int = 1024

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        sequences = [item["sequence"][: self.max_length] for item in batch]
        raw_labels = [item["labels"][: self.max_length] for item in batch]

        encoded = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=self.max_length + 8,  # small slack for special tokens
            return_tensors="pt",
            return_special_tokens_mask=True,
            is_split_into_words=False,
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        special_tokens_mask = encoded["special_tokens_mask"]

        batch_size, seq_len = input_ids.shape
        labels = torch.full((batch_size, seq_len), LABEL_IGNORE_INDEX, dtype=torch.long)

        for i in range(batch_size):
            residue_positions = (special_tokens_mask[i] == 0).nonzero(as_tuple=True)[0]
            this_labels = raw_labels[i]
            n = min(len(residue_positions), len(this_labels))
            if n < len(this_labels):
                # Truncation ate into real residues; only the kept prefix gets labels.
                pass
            positions = residue_positions[:n]
            labels[i, positions] = torch.tensor(this_labels[:n], dtype=torch.long)

        # Padding positions are never real residues, but belt-and-suspenders:
        labels[attention_mask == 0] = LABEL_IGNORE_INDEX

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "ids": [item["id"] for item in batch],
        }


def build_datasets(
    data_dir: str,
    train_file: str,
    valid_file: str,
    test_file: str,
    combine_train_valid_for_final: bool = False,
) -> Tuple[LIPDataset, Optional[LIPDataset], LIPDataset]:
    """Returns (train_dataset, valid_dataset_or_None, test_dataset).

    If combine_train_valid_for_final is True, valid.txt is folded into the
    training set and the returned valid dataset is None (no held-out split
    left to validate against).
    """
    train_path = os.path.join(data_dir, train_file)
    valid_path = os.path.join(data_dir, valid_file)
    test_path = os.path.join(data_dir, test_file)

    if combine_train_valid_for_final:
        train_ds = LIPDataset([train_path, valid_path])
        valid_ds = None
    else:
        train_ds = LIPDataset([train_path])
        valid_ds = LIPDataset([valid_path])

    test_ds = LIPDataset([test_path])
    return train_ds, valid_ds, test_ds