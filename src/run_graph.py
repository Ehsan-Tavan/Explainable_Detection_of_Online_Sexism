# -*- coding: utf-8 -*-

"""
    Explainable Detection of Online Sexism Project:
        src:
            run_graph.py
"""
import logging
# ============================ Third Party libs ============================
import os
import pickle

import numpy as np
import torch
import torch.utils.data as Data
import transformers
from sklearn.utils.class_weight import compute_class_weight

# ============================ My packages ============================
from configuration import BaseConfig
from data_loader import read_csv
from graph_constructor import GraphBuilder, SemanticAdjacencyMatrixBuilder, \
    SyntacticAdjacencyMatrixBuilder, SequentialAdjacencyMatrixBuilder
from models import GCNModel
from utils import IgniteTrainer, normalize_text

# ========================================================================
torch.cuda.empty_cache()

if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()

    # Load data
    print("Loading train data...... ...")
    TRAIN_DATA = read_csv(os.path.join(ARGS.raw_data_dir, ARGS.train_data_file),
                          columns=ARGS.data_headers,
                          names=ARGS.customized_headers)

    TRAIN_DATA.text = TRAIN_DATA.text.apply(lambda x: normalize_text(x))
    print("train set contain %s sample ..." % len(TRAIN_DATA))
    print("Loading test data ...")
    TEST_DATA = read_csv(os.path.join(ARGS.raw_data_dir, ARGS.test_data_file),
                         columns=ARGS.test_data_headers,
                         names=ARGS.test_customized_headers)
    TEST_DATA.text = TEST_DATA.text.apply(lambda x: normalize_text(x))

    print("test set contain %s sample ..." % len(TEST_DATA))

    print("Loading extra data ...")
    # EXTRA_DATA = read_csv(os.path.join(ARGS.raw_data_dir, ARGS.extra_data_file))
    # EXTRA_DATA.text = EXTRA_DATA.text.apply(lambda x: normalize_text(x))
    # EXTRA_DATA = EXTRA_DATA[EXTRA_DATA.text != ""]

    # print("extra set contain %s sample ..." % len(EXTRA_DATA))

    TOKENIZER = transformers.AutoTokenizer.from_pretrained(ARGS.tokenizer_model_path,
                                                           use_fast=False)
    logging.info("loading LM tokenizer ...")
    LM_MODEL = transformers.AutoModel.from_pretrained(ARGS.lm_model_path)
    print("loading LM model ...")

    print("Creating sequential adjacency builder ...")
    SEQUENTIAL_ADJACENCY_BUILDER_OBJ = SequentialAdjacencyMatrixBuilder(
        docs=list(TRAIN_DATA.text) + list(TEST_DATA.text),
        arg=ARGS, logger=logging, sbert_model=LM_MODEL)
    SEQUENTIAL_ADJACENCY_BUILDER_OBJ.setup()
    print("Sequential adjacency setup completed ...")

    print("Creating syntactic adjacency builder ...")
    SYNTACTIC_ADJACENCY_BUILDER_OBJ = SyntacticAdjacencyMatrixBuilder(
        docs=list(TRAIN_DATA.text) + list(TEST_DATA.text),
        arg=ARGS, logger=logging, sbert_model=LM_MODEL)
    SYNTACTIC_ADJACENCY_BUILDER_OBJ.setup()
    print("Syntactic adjacency setup completed ...")

    print("Creating semantic adjacency builder ...")
    SEMANTIC_ADJACENCY_BUILDER_OBJ = SemanticAdjacencyMatrixBuilder(
        docs=list(TRAIN_DATA.text) + list(TEST_DATA.text),
        arg=ARGS, logger=logging, sbert_model=LM_MODEL)
    SEMANTIC_ADJACENCY_BUILDER_OBJ.setup()
    print("Semantic adjacency setup completed ...")

    SEMANTIC_ADJACENCY_BUILDER_OBJ.build_adjacency_matrix()

    if not os.path.exists(ARGS.adjacency_file_path):
        print("Building Adjacency matrices ...")
        INFO_NAME2ADJACENCY_MATRIX = {
            "sequential": SEQUENTIAL_ADJACENCY_BUILDER_OBJ.build_adjacency_matrix(),
            "syntactic": SYNTACTIC_ADJACENCY_BUILDER_OBJ.build_adjacency_matrix(),
            "semantic": SEMANTIC_ADJACENCY_BUILDER_OBJ.build_adjacency_matrix()}

        with open(ARGS.adjacency_file_path, "wb") as outfile:
            pickle.dump(INFO_NAME2ADJACENCY_MATRIX, outfile)
    else:
        print("Loading adjacency matrices ...")
        with open(ARGS.adjacency_file_path, "rb") as file:
            INFO_NAME2ADJACENCY_MATRIX = pickle.load(file)

    print("Creating graph builder ...")
    graph_builder_obj = GraphBuilder(
        info_name2adjacency_matrix=INFO_NAME2ADJACENCY_MATRIX,
        labels=list(TRAIN_DATA.label_sexist),
        nod_id2node_value=SEQUENTIAL_ADJACENCY_BUILDER_OBJ.nod_id2node_value,
        num_test=len(TEST_DATA),
        num_train=len(TRAIN_DATA),
        id2doc=SEQUENTIAL_ADJACENCY_BUILDER_OBJ.index2doc,
        node_feats_path=ARGS.node_features_path,
        max_length=ARGS.max_len
    )

    print("Initializing node features ...")
    graph_builder_obj.init_node_features(mode=ARGS.init_type,
                                         tokenizer=TOKENIZER, bert_model=LM_MODEL)

    print("Creating data masks ...")
    data_masks = graph_builder_obj.split_data(dev_size=ARGS.dev_size)

    print("Creating data masks ...")
    GRAPH = graph_builder_obj.get_pyg_graph(data_masks)

    torch.manual_seed(12345)
    MODEL = GCNModel(bert_model=LM_MODEL,
                     graph=GRAPH,
                     num_feature=1024,
                     hidden_dim=512,
                     num_classes=len(set(graph_builder_obj.labels)) - 2)

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

    IDX_LOADER_TRAIN = Data.DataLoader(train_idx, batch_size=ARGS.train_batch_size, shuffle=True)
    IDX_LOADER_VAL = Data.DataLoader(val_idx, batch_size=ARGS.train_batch_size, shuffle=False)
    IDX_LOADER_TEST = Data.DataLoader(test_idx, batch_size=ARGS.train_batch_size, shuffle=False)
    IDX_LOADER = Data.DataLoader(doc_idx, batch_size=ARGS.train_batch_size, shuffle=True)

    MODEL.to(ARGS.device)
    GRAPH.to(ARGS.device)

    CRITERION = torch.nn.CrossEntropyLoss(weight=torch.Tensor(class_weights)).to(ARGS.device)
    BEST_LOSS = float("inf")
    MODEL_PATH = ""
    BEST_EPOCH = 0
    MIN_EPOCH = 5
    TRAINER = IgniteTrainer(model=MODEL, graph=GRAPH, device=ARGS.device, train_iterator=IDX_LOADER,
                            valid_iterator=IDX_LOADER_VAL, checkpoint_path="../assets/saved_models",
                            class_weights=class_weights)
    TRAINER.setup()
    TRAINER.trainer.run(IDX_LOADER, max_epochs=20)
