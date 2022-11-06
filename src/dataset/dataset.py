# -*- coding: utf-8 -*-
# ========================================================
"""
    Explainable Detection of Online Sexism Project:
        dataset:
            dataset.py
"""
# ============================ Third Party libs ============================
from typing import List
import torch


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, texts: List[str], labels: list, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item_index):
        text = self.texts[item_index]
        label = self.labels[item_index]

        encoded_input = self.tokenizer.encode_plus(text=text,
                                                   add_special_tokens=True,
                                                   max_length=self.max_len,
                                                   return_tensors="pt",
                                                   padding="max_length",
                                                   truncation=True,
                                                   return_token_type_ids=True)

        return {"inputs_ids": encoded_input["input_ids"].flatten(),
                "attention_mask": encoded_input["attention_mask"],
                "labels": torch.tensor(label)}
