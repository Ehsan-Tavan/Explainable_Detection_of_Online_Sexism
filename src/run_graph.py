# -*- coding: utf-8 -*-

"""
    Clustering Project:
        src:
            runner
"""
# ============================ Third Party libs ============================
import os
import numpy as np
import gc
import pickle

import pandas as pd
import torch
import transformers
import torch.utils.data as Data
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sentence_transformers import SentenceTransformer

# ============================ My packages ============================
from configuration import BaseConfig
from data_loader import read_csv
from utils import AdjacencyMatrixBuilder, GraphBuilder, train_step, visualize, test_step, predict
from models import GCNModel

# ========================================================================


if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()

    # Load data
    TRAIN_DATA = read_csv(os.path.join(ARGS.raw_data_dir, ARGS.train_data_file),
                          columns=ARGS.data_headers,
                          names=ARGS.customized_headers)

    TEST_DATA = read_csv(os.path.join(ARGS.raw_data_dir, ARGS.test_data_file),
                         columns=ARGS.test_data_headers,
                         names=ARGS.test_customized_headers)

    SBERT_MODEL = SentenceTransformer(ARGS.sbert_model)
    TOKENIZER = transformers.XLMRobertaTokenizer.from_pretrained(
        "/home/LanguageModels/xlm-roberta-base")

    BERT_MODEL = transformers.XLMRobertaModel.from_pretrained(
        "/home/LanguageModels/xlm-roberta-base")

    ADJACENCY_BUILDER_OBJ = AdjacencyMatrixBuilder(
        docs=list(TRAIN_DATA.text) + list(TEST_DATA.text),
        window_size=ARGS.window_size, args=ARGS)

    if not os.path.exists(ARGS.adjacency_file_path):
        ADJACENCY_MATRIX = ADJACENCY_BUILDER_OBJ.build_adjacency_matrix()

        with open(ARGS.adjacency_file_path, "wb") as outfile:
            pickle.dump(ADJACENCY_MATRIX, outfile)
    else:
        with open(ARGS.adjacency_file_path, "rb") as file:
            ADJACENCY_MATRIX = pickle.load(file)

    graph_builder_obj = GraphBuilder(adjacency_matrix=ADJACENCY_MATRIX,
                                     labels=list(TRAIN_DATA.label_sexist),
                                     nod_id2node_value=ADJACENCY_BUILDER_OBJ.nod_id2node_value,
                                     num_test=len(TEST_DATA),
                                     id2doc=ADJACENCY_BUILDER_OBJ.index2doc)
    graph_builder_obj.init_node_features(mode=ARGS.init_type, sbert_model=SBERT_MODEL,
                                         tokenizer=TOKENIZER, bert_model=BERT_MODEL)
    data_masks = graph_builder_obj.split_data(dev_size=ARGS.dev_size)
    GRAPH = graph_builder_obj.get_pyg_graph(data_masks)

    torch.manual_seed(12345)
    MODEL = GCNModel(bert_model=BERT_MODEL,
                     num_feature=768,
                     hidden_dim=256,
                     num_classes=len(set(graph_builder_obj.labels)) - 2)
    print(MODEL)

    class_weights = compute_class_weight(
        y=np.array(GRAPH.y[GRAPH.train_mask]),
        classes=np.unique(GRAPH.y[GRAPH.train_mask]),
        class_weight="balanced")

    NB_TRAIN, NB_VAL, NB_TEST = GRAPH.train_mask.sum(), GRAPH.val_mask.sum(), GRAPH.test_mask.sum()
    # create index loader
    train_idx = Data.TensorDataset(torch.arange(0, NB_TRAIN, dtype=torch.long))
    val_idx = Data.TensorDataset(torch.arange(NB_TRAIN, NB_TRAIN + NB_VAL, dtype=torch.long))
    test_idx = Data.TensorDataset(
        torch.arange(NB_TRAIN + NB_VAL, NB_TRAIN + NB_VAL + NB_TEST, dtype=torch.long))
    doc_idx = Data.ConcatDataset([train_idx, val_idx, test_idx])

    batch_size = 128
    IDX_LOADER_TRAIN = Data.DataLoader(train_idx, batch_size=batch_size, shuffle=True)
    IDX_LOADER_VAL = Data.DataLoader(val_idx, batch_size=batch_size, shuffle=False)
    IDX_LOADER_TEST = Data.DataLoader(test_idx, batch_size=batch_size, shuffle=False)
    IDX_LOADER = Data.DataLoader(doc_idx, batch_size=batch_size, shuffle=True)

    MODEL.to(ARGS.device)
    GRAPH.to(ARGS.device)

    # OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=0.0001, weight_decay=5e-4)
    OPTIMIZER = torch.optim.Adam([
        {'params': MODEL.lm_model.parameters(), 'lr': 1e-5},
        {'params': MODEL.classifier.parameters(), 'lr': 1e-3},
        {'params': MODEL.gnn_model.parameters(), 'lr': 1e-3},
    ], lr=1e-3
    )

    CRITERION = torch.nn.CrossEntropyLoss(weight=torch.Tensor(class_weights)).to(ARGS.device)
    BEST_LOSS = float("inf")
    MODEL_PATH = ""
    BEST_EPOCH = 0
    MIN_EPOCH = 2
    for epoch in range(100):
        LOSS, ACCURACY = train_step(GRAPH, IDX_LOADER, MODEL, OPTIMIZER, CRITERION, ARGS.device)

        print(f"epoch {epoch + 1}, train acc is : {ACCURACY:.4f}, and train loss is : , {LOSS:.3f}")
        PREDICTIONS, TRUE_LABEL, VAL_LOSS = test_step(GRAPH, IDX_LOADER_VAL, MODEL, CRITERION,
                                                      ARGS.device)
        acc_test = accuracy_score(GRAPH.y[GRAPH.val_mask].cpu(), PREDICTIONS)
        f11_test = f1_score(GRAPH.y[GRAPH.val_mask].cpu(), PREDICTIONS,
                            average="macro")
        f12_test = f1_score(GRAPH.y[GRAPH.val_mask].cpu(), PREDICTIONS,
                            average="weighted")
        print("acc_test: {:.4f}".format(acc_test))
        print("f11_test: {:.4f}".format(f11_test))
        print("f12_test: {:.4f}".format(f12_test))
        print(classification_report(y_true=GRAPH.y[GRAPH.val_mask].cpu(),
                                    y_pred=PREDICTIONS))
        if VAL_LOSS < BEST_LOSS:
            MODEL_PATH = f"../assets/saved_models/model_epoch_{epoch + 1}_val_loss_{VAL_LOSS}.pt"
            torch.save(MODEL, MODEL_PATH)
            BEST_EPOCH = epoch + 1
        if (BEST_EPOCH < epoch + 6) and (BEST_EPOCH > MIN_EPOCH):
            break

    PREDICTIONS, TRUE_LABEL, VAL_LOSS = test_step(GRAPH, IDX_LOADER_VAL, MODEL, CRITERION,
                                                  ARGS.device)

    acc_test = accuracy_score(GRAPH.y[GRAPH.val_mask].cpu(), PREDICTIONS)
    f11_test = f1_score(GRAPH.y[GRAPH.val_mask].cpu(), PREDICTIONS,
                        average="macro")
    f12_test = f1_score(GRAPH.y[GRAPH.val_mask].cpu(), PREDICTIONS,
                        average="weighted")
    print("acc_test: {:.4f}".format(acc_test))
    print("f11_test: {:.4f}".format(f11_test))
    print("f12_test: {:.4f}".format(f12_test))
    print(classification_report(y_true=GRAPH.y[GRAPH.val_mask].cpu(),
                                y_pred=PREDICTIONS))

    MODEL = torch.load(MODEL_PATH).to(ARGS.device())
    TEST_PRED = predict(GRAPH, IDX_LOADER_TEST, MODEL, ARGS.device)
    TEST_PRED = graph_builder_obj.label_encoder.inverse_transform(TEST_PRED)

    RESULT_DATA = pd.DataFrame(
        {"rewire_id": list(TEST_DATA.rewire_id), "label_pred": list(TEST_PRED)})
    RESULT_DATA.to_csv("./result.csv", index=False)
