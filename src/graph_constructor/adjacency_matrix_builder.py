# -*- coding: utf-8 -*-
# ========================================================
"""
    Explainable Detection of Online Sexism Project:
        graph_constructor:
            adjacency_matrix_builder.py
"""

# ============================ Third Party libs ============================
from tqdm import tqdm
from collections import defaultdict
import scipy.sparse as sp
import spacy
import os
from abc import abstractmethod, ABC
from typing import List
import transformer_embedder as tre
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# ============================ My packages ============================
from utils import filtered_infrequent_vocabs, remove_stop_words_from_vocabs, \
    calculate_tf_idf, calculate_pmi
from data_loader import read_pickle, write_pickle


class AdjacencyMatrixBuilder(ABC):
    def __init__(self, docs: List[str], arg, logger, use_lemma: bool = True,
                 remove_stop_words: bool = True, remove_infrequent_vocabs: bool = True,
                 min_occurrence: int = 3, window_size: int = 5, *args, **kwargs):
        self.docs = docs
        self.arg = arg
        self.use_lemma = use_lemma
        self.remove_stop_words = remove_stop_words
        self.remove_infrequent_vocabs = remove_infrequent_vocabs
        self.logger = logger
        self.min_occurrence = min_occurrence
        self.window_size = window_size

        self.nlp = spacy.load(arg.spacy_model_path)

        self.filtered_vocabs = []
        self.word2index = {}
        self.index2word = {}
        self.index2doc = {}
        self.doc2index = {}

    @abstractmethod
    def setup(self):
        """

        Returns:

        """

    def filter_vocabs(self, vocab2frequency: dict) -> None:
        """

        Args:
            vocab2frequency:

        Returns:

        """
        if os.path.exists(self.arg.filtered_vocabs_path):
            self.logger.info("Loading filtered_vocabs ...")
            self.filtered_vocabs = read_pickle(self.arg.filtered_vocabs_path)
        else:
            self.logger.info("Starting filtered_vocabs ...")
            if self.remove_stop_words:
                self.logger.info("Starting remove stop words ...")
                stopwords = list(self.nlp.Defaults.stop_words)
                self.filtered_vocabs = remove_stop_words_from_vocabs(self.filtered_vocabs,
                                                                     stopwords)
            if self.remove_infrequent_vocabs:
                self.logger.info("Starting remove infrequent vocabs ...")
                self.filtered_vocabs = filtered_infrequent_vocabs(
                    vocab2frequency,
                    min_occurrence=self.min_occurrence)
            write_pickle(self.arg.filtered_vocabs_path, self.filtered_vocabs)

    def build_word_counter(self) -> defaultdict:
        """

        Returns:

        """
        if os.path.exists(self.arg.vocab2frequency_path):
            vocab2frequency = read_pickle(self.arg.vocab2frequency_path)
        else:
            vocab2frequency = defaultdict(int)
            for doc in tqdm(self.docs):
                text = self.nlp(doc)
                for word in text:
                    if self.use_lemma:
                        vocab2frequency[word.lemma_.lower()] += 1
                    else:
                        vocab2frequency[word.text.lower()] += 1
            write_pickle(self.arg.vocab2frequency_path, vocab2frequency)

        return vocab2frequency

    def build_word2doc_ids(self) -> defaultdict:
        """

        Returns:

        """
        word2doc_ids = defaultdict(set)
        for i, doc in tqdm(enumerate(self.docs)):
            doc = self.nlp(doc)
            if self.use_lemma:
                words = [tok.lemma_.lower() for tok in doc]
            else:
                words = [tok.text.lower() for tok in doc]
            words = list(filter(lambda vocab: vocab in self.filtered_vocabs, words))
            for word in words:
                word2doc_ids[word].add(i)
        return word2doc_ids

    @staticmethod
    def build_word2docs_frequency(word2doc_ids: dict) -> dict:
        """

        Args:
            word2doc_ids:

        Returns:

        """
        word2docs_frequency = {}
        for word, doc_ids in word2doc_ids.items():
            word2docs_frequency[word] = len(doc_ids)
        return word2docs_frequency

    def build_word2index(self, vocabs: list) -> None:
        """

        Args:
            vocabs:

        Returns:

        """
        self.word2index = {word: i for i, word in enumerate(vocabs)}

    def build_index2word(self, vocabs: list) -> None:
        """

        Args:
            vocabs:

        Returns:

        """
        self.index2word = {i: word for i, word in enumerate(vocabs)}

    def build_doc2index(self) -> None:
        """

        Returns:

        """
        self.doc2index = {doc: i for i, doc in enumerate(self.docs)}

    def build_index2doc(self) -> None:
        """

        Returns:

        """
        self.index2doc = {i: doc for i, doc in enumerate(self.docs)}

    def build_doc_word2frequency(self) -> defaultdict:
        """

        Returns:

        """
        doc_word2frequency = defaultdict(int)
        for i, doc in tqdm(enumerate(self.docs)):
            doc = self.nlp(doc)
            if self.use_lemma:
                words = [tok.lemma_.lower() for tok in doc]
            else:
                words = [tok.text.lower() for tok in doc]
            words = list(filter(lambda vocab: vocab in self.filtered_vocabs, words))
            for word in words:
                word_id = self.word2index[word]
                doc_word2frequency[(i, word_id)] += 1
        return doc_word2frequency

    @abstractmethod
    def build_word_to_word_edge_weight(self, *args, **kwargs):
        """

        Returns:

        """

    def build_word_to_doc_edge_weight(self, doc_word2frequency: dict, word2docs_frequency: dict,
                                      num_docs: int) -> [list, list, list]:
        """

        Args:
            doc_word2frequency:
            word2docs_frequency:
            num_docs:

        Returns:

        """
        rows, columns, weights = [], [], []
        for doc_index, doc in enumerate(self.docs):
            doc = self.nlp(doc)
            if self.use_lemma:
                words = [tok.lemma_.lower() for tok in doc]
            else:
                words = [tok.text.lower() for tok in doc]
            words = set(filter(lambda vocab: vocab in self.filtered_vocabs, words))
            for word in words:
                word_index = self.word2index[word]
                rows.append(doc_index)
                columns.append(num_docs + word_index)
                word_freq = doc_word2frequency[(doc_index, word_index)]
                n_docs_have_word = word2docs_frequency[self.index2word[word_index]]
                tf_idf = calculate_tf_idf(n_docs=len(self.docs),
                                          word_frequency=word_freq,
                                          n_docs_have_word=n_docs_have_word)
                weights.append(tf_idf)
        return rows, columns, weights

    @abstractmethod
    def build_adjacency_matrix(self):
        """

        Returns:

        """

    def build_word_pair2sentence_count(self) -> defaultdict:
        """

        Returns:

        """
        word_pair2sentence_count = defaultdict(int)
        for doc in tqdm(self.docs):
            text = self.nlp(doc)
            for first_index in range(1, len(text)):
                if text[first_index].lemma_.lower() in self.filtered_vocabs:
                    for second_index in range(first_index):
                        if text[second_index].lemma_.lower() in self.filtered_vocabs:
                            word_pair2sentence_count[
                                (self.word2index[text[first_index].lemma_.lower()],
                                 self.word2index[text[second_index].lemma_.lower()])] += 1
                            word_pair2sentence_count[
                                (self.word2index[text[second_index].lemma_.lower()],
                                 self.word2index[text[first_index].lemma_.lower()])] += 1
        return word_pair2sentence_count


class SemanticAdjacencyMatrixBuilder(AdjacencyMatrixBuilder):

    def __init__(self, docs: List[str], arg, logger, lm_model_path, use_lemma: bool = True,
                 remove_stop_words: bool = True, remove_infrequent_vocabs: bool = True,
                 min_occurrence: int = 3, window_size: int = 5, similarity_threshold: float = 0.80):
        super().__init__(docs, arg, logger, use_lemma, remove_stop_words, remove_infrequent_vocabs,
                         min_occurrence, window_size)
        self.docs = docs
        self.arg = arg
        self.logger = logger
        self.tokenizer = tre.Tokenizer(lm_model_path, language=self.arg.spacy_model_path)
        self.model = tre.TransformerEmbedder(lm_model_path, subtoken_pooling="last",
                                             output_layer="last", )
        self.use_lemma = use_lemma
        self.remove_stop_words = remove_stop_words
        self.remove_infrequent_vocabs = remove_infrequent_vocabs
        self.min_occurrence = min_occurrence
        self.similarity_threshold = similarity_threshold

        self.filtered_vocabs = []
        self.word2index = {}
        self.index2word = {}
        self.index2doc = {}
        self.doc2index = {}
        self.nod_id2node_value = {}

    def setup(self):
        self.logger.info("Creating vocab2frequency ...")
        vocab2frequency = self.build_word_counter()
        self.filtered_vocabs = list(vocab2frequency.values())
        self.filter_vocabs(vocab2frequency)
        self.logger.info("Creating nod_id2node_value ...")
        self.nod_id2node_value = {idx: data for idx, data in
                                  enumerate(self.docs + self.filtered_vocabs)}

        self.logger.info("Creating word2index ...")
        self.build_word2index(self.filtered_vocabs)

        self.logger.info("Creating index2word ...")
        self.build_index2word(self.filtered_vocabs)

        self.logger.info("Creating index2word ...")
        self.build_index2doc()

    def build_word_pair2semantic_relation_count(self):
        """

        Returns:

        """
        word_pair2semantic_relation_count = defaultdict(int)
        for doc in tqdm(self.docs):
            tokenized_text = self.tokenizer(doc, return_tensors=True, use_spacy=True)
            outputs = self.model(**tokenized_text).word_embeddings.squeeze()[1:-1].detach().numpy()
            word_similarity_matrix = cosine_similarity(outputs, outputs)
            text = self.nlp(doc)
            assert len(text) == len(outputs)
            for similarity_index, token in enumerate(text):
                if token.lemma_.lower() in self.filtered_vocabs:
                    for token_index, similarity in enumerate(
                            word_similarity_matrix[similarity_index]):
                        if (text[token_index].lemma_.lower() in self.filtered_vocabs) and \
                                (similarity > self.similarity_threshold) and (
                                similarity_index != token_index):
                            word_pair2semantic_relation_count[
                                (self.word2index[token.lemma_.lower()],
                                 self.word2index[text[token_index].lemma_.lower()])] += 1
                            word_pair2semantic_relation_count[
                                (self.word2index[text[token_index].lemma_.lower()],
                                 self.word2index[token.lemma_.lower()])] += 1

        return word_pair2semantic_relation_count

    def build_word_to_word_edge_weight(self, word_pair2semantic_relation_count: dict,
                                       word_pair2sentence_count: dict, num_docs: int) -> [list,
                                                                                          list,
                                                                                          list]:
        """

        Args:
            word_pair2semantic_relation_count:
            word_pair2sentence_count:
            num_docs:

        Returns:

        """
        rows, columns, weights = [], [], []

        for word_pair_ids, word_pair_frequency in tqdm(word_pair2semantic_relation_count.items()):
            first_word_id, second_word_id = word_pair_ids[0], word_pair_ids[1]
            pair_count = word_pair2sentence_count[word_pair_ids]
            rows.append(num_docs + first_word_id)
            columns.append(num_docs + second_word_id)
            syntactic_relation_weight = word_pair_frequency / pair_count
            weights.append(syntactic_relation_weight)
        return rows, columns, weights

    def build_adjacency_matrix(self):
        """

        Returns:

        """
        num_docs = len(self.docs)

        if not os.path.exists(self.arg.semantic_word_to_word_edge_weight_path):
            self.logger.info("Creating semantic_word_to_word_edge_weight ...")
            word_pair2semantic_relation_count = self.build_word_pair2semantic_relation_count()
            word_pair2sentence_count = self.build_word_pair2sentence_count()
            word_to_word_rows, word_to_word_columns, word_to_word_weights = \
                self.build_word_to_word_edge_weight(word_pair2semantic_relation_count,
                                                    word_pair2sentence_count, num_docs)
            with open(self.arg.semantic_word_to_word_edge_weight_path, "wb") as outfile:
                pickle.dump([word_to_word_rows, word_to_word_columns, word_to_word_weights],
                            outfile)
        else:
            with open(self.arg.semantic_word_to_word_edge_weight_path, "rb") as file:
                word_to_word_rows, word_to_word_columns, word_to_word_weights = \
                    pickle.load(file)

        if not os.path.exists(self.arg.semantic_word_to_doc_edge_weight_path):
            self.logger.info("Creating semantic_word_to_doc_edge_weight ...")
            doc_word2frequency = self.build_doc_word2frequency()
            word2doc_ids = self.build_word2doc_ids()
            word2docs_frequency = self.build_word2docs_frequency(word2doc_ids)

            word_to_doc_rows, word_to_doc_columns, word_to_doc_weights = \
                self.build_word_to_doc_edge_weight(doc_word2frequency, word2docs_frequency,
                                                   num_docs)
            with open(self.arg.semantic_word_to_doc_edge_weight_path, "wb") as outfile:
                pickle.dump([word_to_doc_rows, word_to_doc_columns, word_to_doc_weights],
                            outfile)
        else:
            with open(self.arg.semantic_word_to_doc_edge_weight_path, "rb") as file:
                word_to_doc_rows, word_to_doc_columns, word_to_doc_weights = \
                    pickle.load(file)

        rows = word_to_word_rows + word_to_doc_rows
        columns = word_to_word_columns + word_to_doc_columns
        weights = word_to_word_weights + word_to_doc_weights

        number_nodes = num_docs + len(rows)
        adj_mat = sp.csr_matrix((weights, (rows, columns)), shape=(number_nodes, number_nodes))
        adjacency_matrix = adj_mat + adj_mat.T.multiply(adj_mat.T > adj_mat) - adj_mat.multiply(
            adj_mat.T > adj_mat)
        return adjacency_matrix


class SyntacticAdjacencyMatrixBuilder(AdjacencyMatrixBuilder):

    def __init__(self, docs: List[str], arg, logger, use_lemma: bool = True,
                 remove_stop_words: bool = True, remove_infrequent_vocabs: bool = True,
                 min_occurrence: int = 3, window_size: int = 5):
        super().__init__(docs, arg, logger, use_lemma, remove_stop_words, remove_infrequent_vocabs,
                         min_occurrence, window_size)
        self.docs = docs
        self.arg = arg
        self.logger = logger
        self.use_lemma = use_lemma
        self.remove_stop_words = remove_stop_words
        self.remove_infrequent_vocabs = remove_infrequent_vocabs
        self.min_occurrence = min_occurrence
        self.window_size = window_size

        self.filtered_vocabs = []
        self.word2index = {}
        self.index2word = {}
        self.index2doc = {}
        self.doc2index = {}
        self.nod_id2node_value = {}

    def setup(self):
        self.logger.info("Creating vocab2frequency ...")
        vocab2frequency = self.build_word_counter()
        self.filtered_vocabs = list(vocab2frequency.values())
        self.filter_vocabs(vocab2frequency)
        self.logger.info("Creating nod_id2node_value ...")
        self.nod_id2node_value = {idx: data for idx, data in
                                  enumerate(self.docs + self.filtered_vocabs)}

        self.logger.info("Creating word2index ...")
        self.build_word2index(self.filtered_vocabs)
        self.logger.info("Creating index2word ...")
        self.build_index2word(self.filtered_vocabs)
        self.logger.info("Creating index2doc ...")
        self.build_index2doc()

    def build_word_pair2syntactic_relation_count(self):
        """

        Returns:

        """
        word_pair2syntactic_relation_count = defaultdict(int)
        for doc in tqdm(self.docs):
            text = self.nlp(doc)
            for token in text:
                if token.lemma_.lower() in self.filtered_vocabs:
                    for child in token.children:
                        if child.lemma_.lower() in self.filtered_vocabs:
                            word_pair2syntactic_relation_count[
                                (self.word2index[token.lemma_.lower()],
                                 self.word2index[child.lemma_.lower()])] += 1
                            word_pair2syntactic_relation_count[
                                (self.word2index[child.lemma_.lower()],
                                 self.word2index[token.lemma_.lower()])] += 1
        return word_pair2syntactic_relation_count

    def build_word_to_word_edge_weight(self, word_pair2syntactic_relation_count: dict,
                                       word_pair2sentence_count: dict, num_docs: int) -> [list,
                                                                                          list,
                                                                                          list]:
        """

        Args:
            word_pair2syntactic_relation_count:
            word_pair2sentence_count:
            num_docs:

        Returns:

        """
        rows, columns, weights = [], [], []

        for word_pair_ids, word_pair_frequency in tqdm(word_pair2syntactic_relation_count.items()):
            first_word_id, second_word_id = word_pair_ids[0], word_pair_ids[1]
            pair_count = word_pair2sentence_count[word_pair_ids]
            rows.append(num_docs + first_word_id)
            columns.append(num_docs + second_word_id)
            syntactic_relation_weight = word_pair_frequency / pair_count
            weights.append(syntactic_relation_weight)
        return rows, columns, weights

    def build_adjacency_matrix(self):
        """

        Returns:

        """
        num_docs = len(self.docs)

        if not os.path.exists(self.arg.syntactic_word_to_word_edge_weight_path):
            self.logger.info("Creating syntactic_word_to_word_edge_weight ...")
            word_pair2syntactic_relation_count = self.build_word_pair2syntactic_relation_count()
            word_pair2sentence_count = self.build_word_pair2syntactic_relation_count()
            word_to_word_rows, word_to_word_columns, word_to_word_weights = \
                self.build_word_to_word_edge_weight(word_pair2syntactic_relation_count,
                                                    word_pair2sentence_count, num_docs)
            with open(self.arg.syntactic_word_to_word_edge_weight_path, "wb") as outfile:
                pickle.dump([word_to_word_rows, word_to_word_columns, word_to_word_weights],
                            outfile)
        else:
            with open(self.arg.syntactic_word_to_word_edge_weight_path, "rb") as file:
                word_to_word_rows, word_to_word_columns, word_to_word_weights = \
                    pickle.load(file)

        if not os.path.exists(self.arg.syntactic_word_to_doc_edge_weight_path):
            self.logger.info("Creating syntactic_word_to_doc_edge_weight ...")
            doc_word2frequency = self.build_doc_word2frequency()
            word2doc_ids = self.build_word2doc_ids()
            word2docs_frequency = self.build_word2docs_frequency(word2doc_ids)
            word_to_doc_rows, word_to_doc_columns, word_to_doc_weights = \
                self.build_word_to_doc_edge_weight(doc_word2frequency, word2docs_frequency,
                                                   num_docs)
            with open(self.arg.syntactic_word_to_doc_edge_weight_path, "wb") as outfile:
                pickle.dump([word_to_doc_rows, word_to_doc_columns, word_to_doc_weights],
                            outfile)
        else:
            with open(self.arg.syntactic_word_to_doc_edge_weight_path, "rb") as file:
                word_to_doc_rows, word_to_doc_columns, word_to_doc_weights = \
                    pickle.load(file)

        rows = word_to_word_rows + word_to_doc_rows
        columns = word_to_word_columns + word_to_doc_columns
        weights = word_to_word_weights + word_to_doc_weights

        number_nodes = num_docs + len(rows)
        adj_mat = sp.csr_matrix((weights, (rows, columns)), shape=(number_nodes, number_nodes))
        adjacency_matrix = adj_mat + adj_mat.T.multiply(adj_mat.T > adj_mat) - adj_mat.multiply(
            adj_mat.T > adj_mat)
        return adjacency_matrix


class SequentialAdjacencyMatrixBuilder(AdjacencyMatrixBuilder):
    def __init__(self, docs: List[str], arg, logger, use_lemma: bool = True,
                 remove_stop_words: bool = True, remove_infrequent_vocabs: bool = True,
                 min_occurrence: int = 3, window_size: int = 5):
        super().__init__(docs, arg, logger, use_lemma, remove_stop_words, remove_infrequent_vocabs,
                         min_occurrence, window_size)
        self.docs = docs
        self.args = arg
        self.logger = logger
        self.use_lemma = use_lemma
        self.remove_stop_words = remove_stop_words
        self.remove_infrequent_vocabs = remove_infrequent_vocabs
        self.min_occurrence = min_occurrence
        self.window_size = window_size

        self.word2index = {}
        self.index2word = {}
        self.windows = []
        self.index2doc = {}
        self.doc2index = {}
        self.filtered_vocabs = []
        self.nod_id2node_value = {}
        self.nlp = spacy.load(self.args.spacy_model_path)

    def setup(self):
        self.logger.info("Creating vocab2frequency ...")
        vocab2frequency = self.build_word_counter()
        self.filtered_vocabs = list(vocab2frequency.values())
        self.filter_vocabs(vocab2frequency)
        self.logger.info("Creating nod_id2node_value ...")
        self.nod_id2node_value = {idx: data for idx, data in
                                  enumerate(self.docs + self.filtered_vocabs)}
        self.logger.info("Creating word2index ...")
        self.build_word2index(self.filtered_vocabs)
        self.logger.info("Creating index2word ...")
        self.build_index2word(self.filtered_vocabs)
        self.logger.info("Creating index2doc ...")
        self.build_index2doc()
        self.logger.info("Creating windows ...")
        self.build_windows_of_words()

    def build_windows_of_words(self):
        """

        Returns:

        """
        if os.path.exists(self.args.windows_path):
            self.windows = read_pickle(self.args.windows_path)
        else:
            for doc in tqdm(self.docs):
                doc = self.nlp(doc)
                if self.use_lemma:
                    words = [tok.lemma_.lower() for tok in doc]
                else:
                    words = [tok.text.lower() for tok in doc]
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
    def build_word2window_frequency(windows: List[list]) -> defaultdict:
        """

        Args:
            windows:

        Returns:

        """
        word2window_freq = defaultdict(int)
        for window in windows:
            window = set(window)
            for word in window:
                word2window_freq[word] += 1
        return word2window_freq

    def build_word_pair2frequency(self, windows: List[list]) -> defaultdict:
        """

        Args:
            windows:

        Returns:

        """
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

    def build_word_to_word_edge_weight(self, num_docs: int, num_window: int,
                                       word_pair2frequency: dict, word2window_freq: dict) -> [list,
                                                                                              list,
                                                                                              list]:
        """

        Args:
            num_docs:
            num_window:
            word_pair2frequency:
            word2window_freq:

        Returns:

        """
        rows, columns, weights = [], [], []

        for word_pair_ids, word_pair_frequency in tqdm(word_pair2frequency.items()):
            first_word_id, second_word_id = word_pair_ids[0], word_pair_ids[1]
            first_word_frequency = word2window_freq[self.index2word[first_word_id]]
            second_word_frequency = word2window_freq[self.index2word[second_word_id]]
            pmi = calculate_pmi(word_pair_frequency, first_word_frequency,
                                second_word_frequency, num_window)
            if pmi > 0:
                rows.append(num_docs + first_word_id)
                columns.append(num_docs + second_word_id)
                weights.append(pmi)

        return rows, columns, weights

    def build_adjacency_matrix(self):
        """

        Returns:

        """
        num_docs = len(self.docs)

        if not os.path.exists(self.arg.sequential_word_to_word_edge_weight_path):
            self.logger.info("Creating sequential_word_to_word_edge_weight ...")
            word_pair2frequency = self.build_word_pair2frequency(self.windows)
            doc_word2frequency = self.build_doc_word2frequency()
            word2doc_ids = self.build_word2doc_ids()
            word2docs_frequency = self.build_word2docs_frequency(word2doc_ids)
            word2window_frequency = self.build_word2window_frequency(self.windows)

            num_window = len(self.windows)

            word_to_word_rows, word_to_word_columns, word_to_word_weights = \
                self.build_word_to_word_edge_weight(num_docs, num_window, word_pair2frequency,
                                                    word2window_frequency)
            with open(self.arg.sequential_word_to_word_edge_weight_path, "wb") as outfile:
                pickle.dump([word_to_word_rows, word_to_word_columns, word_to_word_weights,
                             doc_word2frequency, word2docs_frequency],
                            outfile)
        else:
            with open(self.arg.sequential_word_to_word_edge_weight_path, "rb") as file:
                word_to_word_rows, word_to_word_columns, word_to_word_weights, doc_word2frequency, \
                word2docs_frequency = pickle.load(file)

        if not os.path.exists(self.arg.sequential_word_to_doc_edge_weight_path):
            self.logger.info("Creating sequential_word_to_doc_edge_weight ...")
            word_to_doc_rows, word_to_doc_columns, word_to_doc_weights = \
                self.build_word_to_doc_edge_weight(doc_word2frequency, word2docs_frequency,
                                                   num_docs)
            with open(self.arg.sequential_word_to_doc_edge_weight_path, "wb") as outfile:
                pickle.dump([word_to_doc_rows, word_to_doc_columns, word_to_doc_weights],
                            outfile)
        else:
            with open(self.arg.sequential_word_to_doc_edge_weight_path, "rb") as file:
                word_to_doc_rows, word_to_doc_columns, word_to_doc_weights = \
                    pickle.load(file)

        rows = word_to_word_rows + word_to_doc_rows
        columns = word_to_word_columns + word_to_doc_columns
        weights = word_to_word_weights + word_to_doc_weights

        number_nodes = num_docs + len(rows)
        adj_mat = sp.csr_matrix((weights, (rows, columns)), shape=(number_nodes, number_nodes))
        adjacency_matrix = adj_mat + adj_mat.T.multiply(adj_mat.T > adj_mat) - adj_mat.multiply(
            adj_mat.T > adj_mat)
        return adjacency_matrix
