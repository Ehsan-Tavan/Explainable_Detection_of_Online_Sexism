# -*- coding: utf-8 -*-
# ========================================================
"""
    Explainable Detection of Online Sexism Project:
        dataset:
            data_module.py
"""
# ============================ Third Party libs ============================
from typing import List
import torch
import pytorch_lightning as pl

# ============================ My packages ============================
from .dataset import CustomDataset


class DataModule(pl.LightningDataModule):
    """
    DataModule
    """

    def __init__(self, train_data: List[str], train_labels: list, val_data: List[str],
                 val_labels: list, config, tokenizer):
        super().__init__()
        self.config = config
        self.train_data = train_data
        self.train_labels = train_labels
        self.val_data = val_data
        self.val_labels = val_labels
        self.tokenizer = tokenizer
        self.customs_dataset = {}

    def setup(self, stage=None) -> None:
        """
        method to setup data module

        Returns:
            None
        """
        self.customs_dataset["train_dataset"] = CustomDataset(
            texts=self.train_data, labels=self.train_labels, tokenizer=self.tokenizer,
            max_len=self.config.max_len)

        self.customs_dataset["val_dataset"] = CustomDataset(
            texts=self.val_data, labels=self.val_labels, tokenizer=self.tokenizer,
            max_len=self.config.max_len)

    def train_dataloader(self):
        """
        method to create train dataloader
        Returns:

        """
        return torch.utils.data.DataLoader(self.customs_dataset["train_dataset"],
                                           batch_size=self.config.train_batch_size,
                                           num_workers=self.config.num_workers)

    def val_dataloader(self):
        """
        method to create validation data loader

        Returns:
            data loader

        """
        return torch.utils.data.DataLoader(self.customs_dataset["val_dataset"],
                                           batch_size=self.config.train_batch_size,
                                           num_workers=self.config.num_workers)

    def test_dataloader(self):
        """
        method to create test data loader

        Returns:
            data loader

        """
        return torch.utils.data.DataLoader(self.customs_dataset["val_dataset"],
                                           batch_size=self.config.train_batch_size,
                                           num_workers=self.config.num_workers)
