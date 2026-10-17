"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset


class PreprocessedDatasetOpenWebText(Dataset):
    def __init__(self, data_dir, block_size=1024, split="train"):
        file_path = f"{split}_qwen.bin"
        self.file_path = file_path
        self.data = np.memmap(
            os.path.join(data_dir, file_path), dtype=np.uint32, mode="r"
        )
        self.block_size = block_size
        self.split = split

        self.data_len = len(self.data) // self.block_size  # drop last block
        self.remain = len(self.data) % self.block_size

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        if self.split == "train":
            if idx < self.data_len - 1:
                random_shift = random.randint(0, self.block_size)
            else:
                random_shift = random.randint(0, self.remain)
            x = torch.from_numpy(
                (
                    self.data[
                        idx * self.block_size
                        + random_shift : (idx + 1) * self.block_size
                        + random_shift
                    ]
                ).astype(np.int64)
            )
        else:
            x = torch.from_numpy(
                (self.data[idx * self.block_size : (idx + 1) * self.block_size]).astype(
                    np.int64
                )
            )
        attention_mask = torch.ones_like(x)
        return {"input_ids": x, "labels": x, "attention_mask": attention_mask}


def get_dataset(cfg, max_seq_len=1024, tokenizer=None):
    if cfg.dataset == "openwebtext" or cfg.dataset == "lm1b":
        train_dataset = PreprocessedDatasetOpenWebText(
            cfg.data_path, block_size=max_seq_len, split="train"
        )
        val_dataset = PreprocessedDatasetOpenWebText(
            cfg.data_path, block_size=max_seq_len, split="val"
        )
        return train_dataset, val_dataset
    elif cfg.dataset == "roc":
        roc_data_path = cfg.data_path
        dataset = load_dataset("text", data_files={f'{split}': os.path.join(roc_data_path, f'roc_{split}.json') for split in ['train', 'valid']})
        dataset = process_roc_dataset(dataset)
        def tokenize(example):
            return tokenizer(example["text"], padding="max_length", truncation=True, max_length=max_seq_len)
        dataset = dataset.map(tokenize, remove_columns='text')
        return dataset['train'], dataset['test']
    else:
        print(f"dataset [{cfg.dataset}] not supported for evaluation")
        raise NotImplementedError

def process_roc_dataset(dataset):
    def extract_roc_text(example):
        text = example['text']
        assert text[:2] == '["'
        assert text[-2:] == '"]'
        sentences = text[2:-2]
        return {'text': sentences}
    dataset = dataset.map(extract_roc_text, )
    dataset = dataset.shuffle(seed=42)
    # Hold out some validation samples for testing
    val_test_ds = dataset['valid'].train_test_split(train_size=1000, shuffle=False)
    dataset['valid'] = val_test_ds['train']
    dataset['test'] = val_test_ds['test']
    return dataset
