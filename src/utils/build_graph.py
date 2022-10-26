# -*- coding: utf-8 -*-
# ========================================================
"""
    Clustering Project:
        data loader:
            data_reader
"""

# ============================ Third Party libs ============================
from tqdm import tqdm
from collections import defaultdict
import scipy.sparse as sp
import spacy
import os

import numpy as np
from typing import List
import torch
from torch_geometric.data import Data as PyGSingleGraphData
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# ============================ My packages ============================
from .helper import create_mask, filtered_infrequent_vocabs, remove_stop_words_from_vocabs
from .helper import calculate_tf_idf, calculate_pmi
from data_loader import read_pickle, write_pickle


class AdjacencyMatrixBuilder:
    def __init__(self, docs: List[str], window_size: int, args):
        self.docs = docs
        self.window_size = window_size
        self.word2index = {}
        self.index2word = {}
        self.windows = []
        self.index2doc = None
        self.filtered_vocabs = []
        self.args = args
        self.nlp = spacy.load('../assets/en_core_web_sm')

        vocab2frequency = self.build_word_counter()
        self.filter_vocabs(vocab2frequency)

        self.nod_id2node_value = {idx: data for idx, data in
                                  enumerate(self.docs + self.filtered_vocabs)}
        self.build_word2index(self.filtered_vocabs)
        self.build_index2word(self.filtered_vocabs)
        self.build_index2doc()
        self.build_windows_of_words()

    def filter_vocabs(self, vocab2frequency):
        if os.path.exists(self.args.filtered_vocabs_path):
            self.filtered_vocabs = read_pickle(self.args.filtered_vocabs_path)
        else:
            vocabs = remove_stop_words_from_vocabs(vocab2frequency.keys())
            self.filtered_vocabs = filtered_infrequent_vocabs(vocabs, vocab2frequency,
                                                              min_occurrence=3)
            write_pickle(self.args.filtered_vocabs_path, self.filtered_vocabs)

    def build_unique_words(self):
        words = set()
        for doc in self.docs:
            words.update(doc.split())
        return list(words)

    def build_word_counter(self):
        if os.path.exists(self.args.vocab2frequency_path):
            vocab2frequency = read_pickle(self.args.vocab2frequency_path)
        else:
            vocab2frequency = defaultdict(int)
            for doc in tqdm(self.docs):
                text = self.nlp(doc)
                for word in text:
                    vocab2frequency[word.lemma_.lower()] += 1
            write_pickle(self.args.vocab2frequency_path, vocab2frequency)

        return vocab2frequency

    def build_word2doc_ids(self):
        word2doc_ids = defaultdict(set)
        for i, doc in tqdm(enumerate(self.docs)):
            doc = self.nlp(doc)
            words = [tok.lemma_.lower() for tok in doc]
            words = list(filter(lambda vocab: vocab in self.filtered_vocabs, words))
            for word in words:
                word2doc_ids[word].add(i)
        return word2doc_ids

    @staticmethod
    def build_word2docs_frequency(word2doc_ids: dict):
        word2docs_frequency = {}
        for word, doc_ids in word2doc_ids.items():
            word2docs_frequency[word] = len(doc_ids)
        return word2docs_frequency

    def build_word2index(self, vocabs: list):
        self.word2index = {word: i for i, word in enumerate(vocabs)}

    def build_index2word(self, vocabs: list):
        self.index2word = {i: word for i, word in enumerate(vocabs)}

    def build_doc2index(self):
        doc2index = {doc: i for i, doc in enumerate(self.docs)}
        return doc2index

    def build_index2doc(self):
        self.index2doc = {i: doc for i, doc in enumerate(self.docs)}

    def build_windows_of_words(self):
        if os.path.exists(self.args.windows_path):
            self.windows = read_pickle(self.args.windows_path)
        else:
            for doc in tqdm(self.docs):
                doc = self.nlp(doc)
                words = [tok.lemma_.lower() for tok in doc]
                doc_length = len(words)
                if doc_length <= self.window_size:
                    self.windows.append(
                        list(filter(lambda vocab: vocab in self.filtered_vocabs, words)))
                else:
                    for idx in range(doc_length - self.window_size + 1):
                        window = words[idx: idx + self.window_size]
                        self.windows.append(
                            list(filter(lambda vocab: vocab in self.filtered_vocabs, window)))
            write_pickle(self.args.windows_path, self.windows)

    @staticmethod
    def build_word2window_frequency(windows):
        word_window_freq = defaultdict(int)
        for window in windows:
            appeared = set()
            for word in window:
                if word not in appeared:
                    word_window_freq[word] += 1
                    appeared.add(word)
        return word_window_freq

    def build_doc_word2frequency(self):
        doc_word2frequency = defaultdict(int)
        for i, doc in tqdm(enumerate(self.docs)):
            doc = self.nlp(doc)
            words = [tok.lemma_.lower() for tok in doc]
            words = list(filter(lambda vocab: vocab in self.filtered_vocabs, words))
            for word in words:
                word_id = self.word2index[word]
                doc_word2frequency[(i, word_id)] += 1
        return doc_word2frequency

    def build_word_pair2frequency(self, windows):
        word_pair2frequency = defaultdict(int)
        for window in tqdm(windows):
            for first_word_index in range(1, len(window)):
                for second_word_index in range(first_word_index):
                    first_word = window[first_word_index]
                    second_word = window[second_word_index]
                    first_word_id = self.word2index[first_word]
                    second_word_id = self.word2index[second_word]
                    if first_word_id == second_word_id:
                        continue
                    word_pair2frequency[(first_word_id, second_word_id)] += 1
                    word_pair2frequency[(second_word_id, first_word_id)] += 1
        return word_pair2frequency

    def build_word_to_word_edge_weight(self, num_docs, num_window, word_pair2frequency,
                                       word_window_freq):
        rows, columns, weights = [], [], []

        for word_pair_ids, word_pair_frequency in tqdm(word_pair2frequency.items()):
            first_word_id, second_word_id = word_pair_ids[0], word_pair_ids[1]
            first_word_frequency = word_window_freq[self.index2word[first_word_id]]
            second_word_frequency = word_window_freq[self.index2word[second_word_id]]
            pmi = calculate_pmi(word_pair_frequency, first_word_frequency,
                                second_word_frequency, num_window)
            if pmi > 0:
                rows.append(num_docs + first_word_id)
                columns.append(num_docs + second_word_id)
                weights.append(pmi)
        return rows, columns, weights

    def build_word_to_doc_edge_weight(self, doc_word2frequency, word2docs_frequency, num_docs):
        rows, columns, weights = [], [], []
        for doc_index, doc in enumerate(self.docs):
            doc = self.nlp(doc)
            words = [tok.lemma_.lower() for tok in doc]
            words = list(filter(lambda vocab: vocab in self.filtered_vocabs, words))
            doc_word_set = set()
            for word in words:
                if word not in doc_word_set:
                    word_index = self.word2index[word]
                    rows.append(doc_index)
                    columns.append(num_docs + word_index)
                    word_freq = doc_word2frequency[(doc_index, word_index)]
                    n_docs_have_word = word2docs_frequency[self.index2word[word_index]]
                    tf_idf = calculate_tf_idf(n_docs=len(self.docs),
                                              word_frequency=word_freq,
                                              n_docs_have_word=n_docs_have_word)
                    weights.append(tf_idf)
                    doc_word_set.add(word)
        return rows, columns, weights

    def build_adjacency_matrix(self):
        num_docs = len(self.docs)
        word_pair2frequency = self.build_word_pair2frequency(self.windows)
        doc_word2frequency = self.build_doc_word2frequency()
        word2doc_ids = self.build_word2doc_ids()
        word2docs_frequency = self.build_word2docs_frequency(word2doc_ids)
        word2window_frequency = self.build_word2window_frequency(self.windows)

        num_window = len(self.windows)

        word_to_word_rows, word_to_word_columns, word_to_word_weights = \
            self.build_word_to_word_edge_weight(num_docs, num_window, word_pair2frequency,
                                                word2window_frequency)

        word_to_doc_rows, word_to_doc_columns, word_to_doc_weights = \
            self.build_word_to_doc_edge_weight(doc_word2frequency, word2docs_frequency, num_docs)

        rows = word_to_word_rows + word_to_doc_rows
        columns = word_to_word_columns + word_to_doc_columns
        weights = word_to_word_weights + word_to_doc_weights

        number_nodes = num_docs + len(self.word2index)
        adj_mat = sp.csr_matrix((weights, (rows, columns)), shape=(number_nodes, number_nodes))
        adjacency_matrix = adj_mat + adj_mat.T.multiply(adj_mat.T > adj_mat) - adj_mat.multiply(
            adj_mat.T > adj_mat)
        return adjacency_matrix


class GraphBuilder:
    def __init__(self, adjacency_matrix, labels: list, nod_id2node_value: dict, num_test: int,
                 id2doc: dict):
        self.adjacency_matrix = adjacency_matrix
        self.nod_id2node_value = nod_id2node_value
        self.labels = labels
        self.num_test = num_test
        self.id2doc = id2doc
        self.node_feats = None
        self.label_encoder = None
        self.input_ids = None
        self.attention_mask = None

        self.labels.extend((["test_data"] * self.num_test))

        if len(self.labels) != len(nod_id2node_value):
            self.labels.extend(["words"] * (len(nod_id2node_value) - len(id2doc)))
        self.make_labels()

    def make_labels(self):
        self.label_encoder = preprocessing.LabelEncoder()
        self.label_encoder.fit(self.labels)

    def init_node_features(self, mode: str, sbert_model=None, tokenizer=None, bert_model=None):
        if mode == "one_hot_init":
            num_nodes = self.adjacency_matrix.shape[0]
            identity = sp.identity(num_nodes)
            ind0, ind1, values = sp.find(identity)
            inds = np.stack((ind0, ind1), axis=0)

            self.node_feats = torch.sparse_coo_tensor(inds, values, dtype=torch.float)
        elif mode == "sbert":
            self.node_feats = torch.tensor(
                sbert_model.encode(list(self.nod_id2node_value.values())))

            # self.node_feats = torch.tensor(TSNE(n_components=2).fit_transform(
            #     self.node_feats.detach().cpu().numpy()))
        elif mode == "bert":
            text = list(self.nod_id2node_value.values())
            tokenized = tokenizer.batch_encode_plus(text, max_length=80, padding="max_length",
                                                    truncation=True, add_special_tokens=True,
                                                    return_tensors="pt")
            self.input_ids = tokenized["input_ids"]
            self.attention_mask = tokenized["attention_mask"]
            self.node_feats = torch.rand(len(self.nod_id2node_value),
                                         768).clone().detach().requires_grad_(True)

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
        adjacency_matrix = self.adjacency_matrix
        adj = adjacency_matrix.tocoo()
        row = torch.from_numpy(adj.row).to(torch.long)
        col = torch.from_numpy(adj.col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)
        edge_weight = torch.from_numpy(adj.data).to(torch.float)
        labels = self.label_encoder.transform(self.labels)
        pyg_graph = PyGSingleGraphData(x=self.node_feats, edge_attr=edge_weight,
                                       y=torch.tensor(labels),
                                       input_ids=self.input_ids,
                                       attention_mask=self.attention_mask,
                                       train_mask=torch.tensor(data_masks[0]),
                                       val_mask=torch.tensor(data_masks[1]),
                                       test_mask=torch.tensor(data_masks[2]),
                                       edge_index=edge_index)
        return pyg_graph
