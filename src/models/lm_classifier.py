# -*- coding: utf-8 -*-
# ========================================================
"""
    Explainable Detection of Online Sexism Project:
        models:
            lm_classifier.py
"""

# ============================ Third Party libs ============================
import torch
import pytorch_lightning as pl
from transformers import AutoModel
from torchmetrics import F1, Accuracy


class Classifier(pl.LightningModule):
    """
    lm classifier
    """

    def __init__(self, lm_model_path, num_classes, class_weights, lr=2e-5):
        """

        Args:
            num_classes: the number of labels (here 1, 0)
            class_weights: the number of labels (here 1, 0)
        """
        super().__init__()

        self.model = AutoModel.from_pretrained(lm_model_path)
        self.classifier = torch.nn.Linear(self.model.config.hidden_size, num_classes)
        self.loss = torch.nn.CrossEntropyLoss(weight=class_weights)
        self.lr = lr

        self.accuracy = Accuracy()
        self.F_score = F1(average="none", num_classes=num_classes)
        self.F_score_total = F1(average="weighted", num_classes=num_classes)

        self.save_hyperparameters()

    def forward(self, batch):
        """
        the input batch going to be processed by forward architecture
        Args:
            batch: the input batch containing data and labels

        Returns: final output vectors

        """

        model_output = self.model(
            input_ids=batch["inputs_ids"],
            attention_mask=batch["attention_mask"]).pooler_output
        return self.classifier(model_output)

    def training_step(self, batch, _):
        """
        Module evaluate models outputs in training phase
        Args:
            batch: input batch to processed with
            _: batch idx

        Returns: dictionary contains 'loss', 'predictions', and 'labels'
        """
        label = batch["labels"].flatten()
        outputs = self.forward(batch)
        loss = self.loss(outputs, label)

        metric2value = {"train_loss": loss,
                        "train_acc": self.accuracy(torch.softmax(outputs, dim=1), label),
                        "train_f1_first_class": self.F_score(torch.softmax(outputs, dim=1), label)[
                            0],
                        "train_f1_second_class": self.F_score(torch.softmax(outputs, dim=1), label)[
                            1],
                        "train_total_F1": self.F_score_total(torch.softmax(outputs, dim=1), label)}

        self.log_dict(metric2value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": label}

    def validation_step(self, batch, _):
        """
        Module evaluate models outputs in valid phase
        Args:
            batch: input batch to processed with
            _: batch idx

        Returns: dictionary contains 'loss', 'predictions', and 'labels'
        """
        label = batch["labels"].flatten()
        outputs = self.forward(batch)
        loss = self.loss(outputs, label)

        metric2value = {"val_loss": loss,
                        "val_acc": self.accuracy(torch.softmax(outputs, dim=1), label),
                        "val_f1_first_class": self.F_score(torch.softmax(outputs, dim=1), label)[0],
                        "val_f1_second_class": self.F_score(torch.softmax(outputs, dim=1), label)[
                            1],
                        "val_total_F1": self.F_score_total(torch.softmax(outputs, dim=1), label)}

        self.log_dict(metric2value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, _):
        """
        Module evaluate models outputs in test phase
        Args:
            batch: input batch to processed with
            _: batch idx

        Returns: dictionary contains 'loss', 'predictions', and 'labels'
        """
        label = batch["labels"].flatten()
        outputs = self.forward(batch)
        loss = self.loss(outputs, label)

        metric2value = {"test_loss": loss,
                        "test_acc": self.accuracy(torch.softmax(outputs, dim=1), label),
                        "test_f1_first_class": self.F_score(torch.softmax(outputs, dim=1), label)[
                            0],
                        "test_f1_second_class": self.F_score(torch.softmax(outputs, dim=1), label)[
                            1],
                        "test_total_F1": self.F_score_total(torch.softmax(outputs, dim=1), label)}

        self.log_dict(metric2value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        """
        Module defines optimizer
        Returns: optimizer

        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return [optimizer]
