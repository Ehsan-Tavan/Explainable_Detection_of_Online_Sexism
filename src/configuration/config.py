# -*- coding: utf-8 -*-
# ========================================================

"""
    Explainable Detection of Online Sexism Project:
        configuration:
                config.py
"""

# ============================ Third Party libs ============================
import argparse
from pathlib import Path

import torch


# ========================================================


class BaseConfig:
    """
        BaseConfig:
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--model_name", type=str, default="Bert")
        self.parser.add_argument("--data_headers", type=list,
                                 default=["rewire_id", "text", "label_sexist", "label_category",
                                          "label_vector"])
        self.parser.add_argument("--customized_headers", type=list,
                                 default=["rewire_id", "text", "label_sexist", "label_category",
                                          "label_vector"])

        self.parser.add_argument("--test_data_headers", type=list,
                                 default=["rewire_id", "text"])
        self.parser.add_argument("--test_customized_headers", type=list,
                                 default=["rewire_id", "text"])

        self.parser.add_argument("--window_size", type=int, default=10)
        self.parser.add_argument("--max_len", type=int, default=50)
        self.parser.add_argument("--dev_size", type=float, default=0.15)
        self.parser.add_argument("--tvt_list", type=list, default=["train", "test", "val"])
        self.parser.add_argument("--random_seed", type=int, default=3)
        self.parser.add_argument("--use_lemma", type=list, default=True)
        self.parser.add_argument("--remove_stop_words", type=bool, default=True)
        self.parser.add_argument("--remove_infrequent_vocabs", type=bool, default=True)
        self.parser.add_argument("--min_occurrence", type=int, default=3)
        self.parser.add_argument("--init_type", type=str, default="bert")
        self.parser.add_argument("--lr", type=float, default=2e-5)
        self.parser.add_argument("--similarity_threshold", type=float, default=0.80)
        self.parser.add_argument("--n_epochs", type=int, default=20)
        self.parser.add_argument("--train_batch_size", type=int, default=128)
        self.parser.add_argument("--dropout", type=float, default=0.3)
        self.parser.add_argument("--num_workers", type=int, default=8)
        self.parser.add_argument("--device", type=str, default=torch.device(
            "cuda:1" if torch.cuda.is_available() else "cpu"), help="")

    def add_path(self) -> None:
        """
        function to add path

        Returns:
            None

        """
        graph_file = "bertweet_window_size_10_min_occurrence_3_use_lemma_remove_stop_words"

        self.parser.add_argument("--raw_data_dir", type=str,
                                 default=Path(__file__).parents[
                                             2].__str__() + "/data/Raw/starting_ki")
        self.parser.add_argument("--assets_dir", type=str, default=Path(__file__).parents[
                                                                       2].__str__() + "/assets")
        self.parser.add_argument("--train_data_file", type=str, default="train_all_tasks.csv")
        self.parser.add_argument("--test_data_file", type=str, default="dev_task_a_entries.csv")
        self.parser.add_argument("--extra_data_file", type=str, default="extra_data.csv")
        self.parser.add_argument("--vocab2frequency_path", type=str,
                                 default=f"../assets/{graph_file}/vocab2frequency.pkl")
        self.parser.add_argument("--adjacency_file_path", type=str,
                                 default=f"../assets/{graph_file}/adjacency_matrix.pkl")
        self.parser.add_argument("--filtered_vocabs_path", type=str,
                                 default=f"../assets/{graph_file}/filtered_vocabs.pkl")

        self.parser.add_argument("--windows_path", type=str,
                                 default=f"../assets/{graph_file}/windows.pkl")

        self.parser.add_argument("--semantic_word_to_word_edge_weight_path", type=str,
                                 default=f"../assets/{graph_file}/"
                                         f"semantic_word_to_word_edge_weight.pkl")
        self.parser.add_argument("--semantic_word_to_doc_edge_weight_path", type=str,
                                 default=f"../assets/{graph_file}/"
                                         f"semantic_word_to_doc_edge_weight.pkl")
        self.parser.add_argument("--syntactic_word_to_word_edge_weight_path", type=str,
                                 default=f"../assets/{graph_file}/"
                                         f"syntactic_word_to_word_edge_weight.pkl")
        self.parser.add_argument("--syntactic_word_to_doc_edge_weight_path", type=str,
                                 default=f"../assets/{graph_file}/"
                                         f"syntactic_word_to_doc_edge_weight.pkl")
        self.parser.add_argument("--sequential_word_to_word_edge_weight_path", type=str,
                                 default=f"../assets/{graph_file}/"
                                         f"sequential_word_to_word_edge_weight.pkl")
        self.parser.add_argument("--sequential_word_to_doc_edge_weight_path", type=str,
                                 default=f"../assets/{graph_file}/"
                                         f"sequential_word_to_doc_edge_weight.pkl")
        self.parser.add_argument("--node_features_path", type=str,
                                 default=f"../assets/{graph_file}/node_feats.pkl")

        self.parser.add_argument("--sbert_model", type=str,
                                 default="../assets/pretrained_models/"
                                         "distiluse-base-multilingual-cased-v2")
        self.parser.add_argument("--lm_model_path", type=str,
                                 default="/home/LanguageModels/bertweet")
        self.parser.add_argument("--tokenizer_model_path", type=str,
                                 default="/home/LanguageModels/bertweet")
        self.parser.add_argument("--spacy_model_path", type=str,
                                 default="../assets/en_core_web_sm")
        self.parser.add_argument("--saved_model_dir", type=str,
                                 default=Path(__file__).parents[
                                             2].__str__() + "/assets/saved_models/")

    def get_config(self):
        """

        :return:
        """
        self.add_path()
        return self.parser.parse_args()
