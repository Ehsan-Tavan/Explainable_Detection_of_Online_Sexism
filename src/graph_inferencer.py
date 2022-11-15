# -*- coding: utf-8 -*-

"""
    Explainable Detection of Online Sexism Project:
        src:
            graph_inferencer.py
"""
# ============================ Third Party libs ============================
import os
import logging
import pandas
import torch.utils.data as Data
import torch
import numpy
import transformers
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
# ============================ My packages ============================
from configuration import BaseConfig
from data_loader import read_pickle, read_csv
from models.lm_classifier import Classifier
from dataset import InferenceDataset

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()

    TEST_DATA = read_csv(os.path.join(ARGS.raw_data_dir, ARGS.test_data_file),
                         columns=ARGS.test_data_headers,
                         names=ARGS.test_customized_headers)

    graph_path = "/home/ehsan.tavan/repetitive-news/assets/saved_models/assets/saved_models/gnn/v2/" \
                 "model_epoch_2_graph.pt"
    model_path = "/home/ehsan.tavan/repetitive-news/assets/saved_models/assets/saved_models/gnn/v2/" \
                 "model_epoch_2_val_f1_score_0.8121.pt"

    LABEL_ENCODER = read_pickle(
        path=os.path.join(ARGS.saved_model_dir, ARGS.model_name, "label_encoder.pkl"))

    graph = torch.load(graph_path)
    model = torch.load(model_path)

    NB_TRAIN, NB_VAL, NB_TEST = graph.train_mask.sum(), graph.val_mask.sum(), graph.test_mask.sum()
    # create index loader
    train_idx = Data.TensorDataset(torch.arange(0, NB_TRAIN, dtype=torch.long))
    val_idx = Data.TensorDataset(torch.arange(NB_TRAIN, NB_TRAIN + NB_VAL, dtype=torch.long))
    test_idx = Data.TensorDataset(
        torch.arange(NB_TRAIN + NB_VAL, NB_TRAIN + NB_VAL + NB_TEST, dtype=torch.long))

    IDX_LOADER_TRAIN = Data.DataLoader(train_idx, batch_size=ARGS.train_batch_size, shuffle=False)
    IDX_LOADER_VAL = Data.DataLoader(val_idx, batch_size=ARGS.train_batch_size, shuffle=False)
    IDX_LOADER_TEST = Data.DataLoader(test_idx, batch_size=ARGS.train_batch_size, shuffle=False)

    PREDICTED_LABELS = []
    for idx in IDX_LOADER_VAL:
        with torch.no_grad():
            PREDICTED_LABELS.extend(torch.argmax(model(graph, idx), dim=1).cpu().detach().numpy())
    # PREDICTED_LABELS = list(LABEL_ENCODER.inverse_transform(PREDICTED_LABELS))

    print(accuracy_score(graph.y[graph.val_mask].cpu().detach().numpy(), list(PREDICTED_LABELS)))
    # RESULTS = pandas.DataFrame(
    #     {"rewire_id": list(TEST_DATA["rewire_id"]), "label_pred": PREDICTED_LABELS})
    #
    # RESULTS.to_csv("result.csv", index=False, encoding="utf-8")

