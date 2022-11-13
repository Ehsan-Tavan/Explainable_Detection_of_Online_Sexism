# -*- coding: utf-8 -*-
# ========================================================
"""
    Explainable Detection of Online Sexism Project:
        graph_constructor:
            graph_constructor.py
"""

# ============================ Third Party libs ============================
import os
import scipy.sparse as sp
import numpy as np
import torch
from torch_geometric.data import Data as PyGSingleGraphData
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# ============================ My packages ============================
from utils import create_mask
from data_loader import read_pickle, write_pickle


class GraphBuilder:
    def __init__(self, info_name2adjacency_matrix: dict, labels: list, nod_id2node_value: dict,
                 num_test: int, id2doc: dict, node_feats_path: str):
        self.info_name2adjacency_matrix = info_name2adjacency_matrix
        self.nod_id2node_value = nod_id2node_value
        self.labels = labels
        self.num_test = num_test
        self.id2doc = id2doc
        self.node_feats = None
        self.label_encoder = None
        self.input_ids = None
        self.attention_mask = None
        self.node_feats_path = node_feats_path

        self.labels.extend((["test_data"] * self.num_test))

        if len(self.labels) != len(nod_id2node_value):
            self.labels.extend(["words"] * (len(nod_id2node_value) - len(id2doc)))
        self.make_labels()

    def make_labels(self):
        self.label_encoder = preprocessing.LabelEncoder()
        self.label_encoder.fit(self.labels)

    def init_node_features(self, mode: str, sbert_model=None, tokenizer=None, bert_model=None):
        if mode == "one_hot_init":
            num_nodes = list(self.info_name2adjacency_matrix.values())[0].shape[0]
            identity = sp.identity(num_nodes)
            ind0, ind1, values = sp.find(identity)
            inds = np.stack((ind0, ind1), axis=0)

            self.node_feats = torch.sparse_coo_tensor(inds, values, dtype=torch.float)
        elif mode == "sbert":
            self.node_feats = torch.tensor(
                sbert_model.encode(list(self.nod_id2node_value.values())))
        elif mode == "bert":
            if os.path.exists(self.node_feats_path):
                self.node_feats, self.input_ids, self.attention_mask = read_pickle(
                    self.node_feats_path)
            else:
                text = list(self.nod_id2node_value.values())
                tokenized = tokenizer.batch_encode_plus(text, max_length=80, padding="max_length",
                                                        truncation=True, add_special_tokens=True,
                                                        return_tensors="pt")
                self.input_ids = tokenized["input_ids"]
                self.attention_mask = tokenized["attention_mask"]
                bert_model.eval()
                with torch.no_grad():
                    lm_output = bert_model(input_ids=self.input_ids,
                                           attention_mask=self.attention_mask).pooler_output
                    self.node_feats = lm_output.clone().detach().requires_grad_(True)
                write_pickle(self.node_feats_path,
                             [self.node_feats, self.input_ids, self.attention_mask])

        else:
            raise NotImplementedError

    def split_data(self, dev_size: float, seed: int = 1234):
        """

        Args:
            dev_size:
            seed:

        Returns:

        """
        doc_id_chunks = train_test_split(list(self.id2doc.keys())[:-self.num_test],
                                         test_size=dev_size, shuffle=False,
                                         random_state=seed)
        doc_id_chunks.append(list(self.id2doc.keys())[-self.num_test:])
        data_masks = create_mask(len(self.nod_id2node_value), doc_id_chunks)
        return data_masks

    def get_pyg_graph(self, data_masks):
        labels = self.label_encoder.transform(self.labels)
        pyg_graph = PyGSingleGraphData(x=self.node_feats,
                                       y=torch.tensor(labels),
                                       input_ids=self.input_ids,
                                       attention_mask=self.attention_mask,
                                       train_mask=torch.tensor(data_masks[0]),
                                       val_mask=torch.tensor(data_masks[1]),
                                       test_mask=torch.tensor(data_masks[2]))

        for info_name, adjacency_matrix in self.info_name2adjacency_matrix.items():
            adj = adjacency_matrix.tocoo()
            row = torch.from_numpy(adj.row).to(torch.long)
            col = torch.from_numpy(adj.col).to(torch.long)
            if "sequential" in info_name:
                sequential_edge_index = torch.stack([row, col], dim=0)
                sequential_edge_weight = torch.from_numpy(adj.data).to(torch.float)
                pyg_graph.sequential_edge_attr = sequential_edge_weight
                pyg_graph.sequential_edge_index = sequential_edge_index

            elif "syntactic" in info_name:
                syntactic_edge_index = torch.stack([row, col], dim=0)
                syntactic_edge_weight = torch.from_numpy(adj.data).to(torch.float)
                pyg_graph.syntactic_edge_attr = syntactic_edge_weight
                pyg_graph.syntactic_edge_index = syntactic_edge_index

            elif "semantic" in info_name:
                semantic_edge_index = torch.stack([row, col], dim=0)
                semantic_edge_weight = torch.from_numpy(adj.data).to(torch.float)
                pyg_graph.semantic_edge_attr = semantic_edge_weight
                pyg_graph.semantic_edge_index = semantic_edge_index

        return pyg_graph
