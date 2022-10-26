# -*- coding: utf-8 -*-
# ========================================================
"""
    Clustering Project:
        data loader:
            data_reader
"""

# ============================ Third Party libs ============================
from typing import List
from collections import defaultdict
from math import log
from tqdm import tqdm
import scipy.sparse as sp


def get_vocab_and_vocab_freq(docs: List[str]) -> [dict, list]:
    """

    Args:
        docs:

    Returns:

    """
    vocab_freq = defaultdict(int)
    for doc in docs:
        words = doc.split()
        for word in words:
            vocab_freq[word] += 1
    return vocab_freq, list(vocab_freq.keys())


def build_word_doc_edges(docs: List[str]) -> [defaultdict, dict]:
    """

    Args:
        docs:

    Returns:

    """
    words_in_docs = defaultdict(set)
    for i, doc in enumerate(docs):
        words = doc.split()
        for word in words:
            words_in_docs[word].add(i)

    word_doc_freq = {}
    for word, doc_ids in words_in_docs.items():
        word_doc_freq[word] = len(doc_ids)
    return words_in_docs, word_doc_freq


def build_windows(docs: List[str], window_size=20) -> List[list]:
    """

    Args:
        docs:
        window_size:

    Returns:

    """
    windows = []
    for doc in docs:
        words = doc.split()
        doc_length = len(words)
        if doc_length <= window_size:
            windows.append(words)
        else:
            for idx in range(doc_length - window_size + 1):
                window = words[idx: idx + window_size]
                windows.append(window)
    return windows


def get_word_freq_in_windows(windows: List[list]) -> defaultdict:
    """

    Args:
        windows:

    Returns:

    """
    word_window_freq = defaultdict(int)

    for window in windows:
        appeared = set()
        for word in window:
            if word not in appeared:
                word_window_freq[word] += 1
                appeared.add(word)
    return word_window_freq


def get_word_pair_count_from_windows(windows: List[list], word2id: dict) -> defaultdict:
    """

    Args:
        windows:
        word2id:

    Returns:

    """
    word_pair_count = defaultdict(int)
    for window in tqdm(windows):
        for i in range(1, len(window)):
            for j in range(i):
                word_i = window[i]
                word_j = window[j]
                word_i_id = word2id[word_i]
                word_j_id = word2id[word_j]
                if word_i_id == word_j_id:
                    continue
                word_pair_count[(word_i_id, word_j_id)] += 1
                word_pair_count[(word_j_id, word_i_id)] += 1
    return word_pair_count


def get_doc_word_frequency(docs: List[str], word2id: dict) -> dict:
    """

    Args:
        docs:
        word2id:

    Returns:

    """
    doc_word_freq = defaultdict(int)
    for i, doc_words in enumerate(docs):
        words = doc_words.split()
        for word in words:
            word_id = word2id[word]
            doc_word_freq[(i, word_id)] += 1
    return doc_word_freq


def build_adjacency_matrix(doc_word_freq: dict, docs: list, vocabs: list, windows: list,
                           word2id: dict, word_doc_freq: dict, word_pair_count: dict,
                           word_window_freq: dict):
    """

    Args:
        doc_word_freq:
        docs:
        vocabs:
        windows:
        word2id:
        word_doc_freq:
        word_pair_count:
        word_window_freq:

    Returns:

    """
    row, col, weight = [], [], []
    # pmi as weights
    num_docs = len(docs)
    num_window = len(windows)
    for word_id_pair, count in tqdm(word_pair_count.items()):
        i, j = word_id_pair[0], word_id_pair[1]
        word_freq_i = word_window_freq[vocabs[i]]
        word_freq_j = word_window_freq[vocabs[j]]
        pmi = log((1.0 * count / num_window) /
                  (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
        if pmi <= 0:
            continue
        row.append(num_docs + i)
        col.append(num_docs + j)
        weight.append(pmi)
    for i, doc in enumerate(docs):
        words = doc.split()
        doc_word_set = set()
        for word in words:
            if word in doc_word_set:
                continue
            word_id = word2id[word]
            freq = doc_word_freq[(i, word_id)]
            row.append(i)
            col.append(num_docs + word_id)
            idf = log(1.0 * num_docs /
                      word_doc_freq[vocabs[word_id]])
            weight.append(freq * idf)
            doc_word_set.add(word)
    number_nodes = num_docs + len(vocabs)
    nod_id2node_value = {idx: data for idx, data in enumerate(docs + vocabs)}
    nodes_value = list(docs + vocabs)
    adj_mat = sp.csr_matrix((weight, (row, col)), shape=(number_nodes, number_nodes))
    adj = adj_mat + adj_mat.T.multiply(adj_mat.T > adj_mat) - adj_mat.multiply(adj_mat.T > adj_mat)
    return adj, nod_id2node_value, nodes_value
