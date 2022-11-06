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

        self.parser.add_argument("--window_size", type=int, default=5)
        self.parser.add_argument("--max_len", type=int, default=100)
        self.parser.add_argument("--dev_size", type=float, default=0.2)
        self.parser.add_argument("--tvt_list", type=list, default=["train", "test", "val"])
        self.parser.add_argument("--random_seed", type=int, default=3)
        self.parser.add_argument("--use_lemma", type=bool, default=True)
        self.parser.add_argument("--remove_stop_words", type=bool, default=True)
        self.parser.add_argument("--remove_infrequent_vocabs", type=bool, default=True)
        self.parser.add_argument("--min_occurrence", type=int, default=3)
        self.parser.add_argument("--init_type", type=str, default="bert")
        self.parser.add_argument("--lr", type=float, default=2e-5)
        self.parser.add_argument("--n_epochs", type=int, default=20)
        self.parser.add_argument("--train_batch_size", type=int, default=64)
        self.parser.add_argument("--num_workers", type=int, default=8)
        self.parser.add_argument("--device", type=str, default=torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"), help="")

    def add_path(self) -> None:
        """
        function to add path

        Returns:
            None

        """

        self.parser.add_argument("--raw_data_dir", type=str,
                                 default=Path(__file__).parents[
                                             2].__str__() + "/data/Raw/starting_ki")

        self.parser.add_argument("--train_data_file", type=str, default="train_all_tasks.csv")
        self.parser.add_argument("--test_data_file", type=str, default="dev_task_a_entries.csv")
        self.parser.add_argument("--vocab2frequency_path", type=str,
                                 default="../assets/vocab2frequency.pkl")
        self.parser.add_argument("--adjacency_file_path", type=str,
                                 default="../assets/adjacency_matrix.pkl")
        self.parser.add_argument("--filtered_vocabs_path", type=str,
                                 default="../assets/filtered_vocabs.pkl")
        self.parser.add_argument("--windows_path", type=str,
                                 default="../assets/windows.pkl")
        self.parser.add_argument("--semantic_word_to_word_edge_weight_path", type=str,
                                 default="../assets/semantic_word_to_word_edge_weight.pkl")
        self.parser.add_argument("--semantic_word_to_doc_edge_weight_path", type=str,
                                 default="../assets/semantic_word_to_doc_edge_weight.pkl")
        self.parser.add_argument("--syntactic_word_to_word_edge_weight_path", type=str,
                                 default="../assets/syntactic_word_to_word_edge_weight.pkl")
        self.parser.add_argument("--syntactic_word_to_doc_edge_weight_path", type=str,
                                 default="../assets/syntactic_word_to_doc_edge_weight.pkl")
        self.parser.add_argument("--sequential_word_to_word_edge_weight_path", type=str,
                                 default="../assets/sequential_word_to_word_edge_weight.pkl")
        self.parser.add_argument("--sequential_word_to_doc_edge_weight_path", type=str,
                                 default="../assets/sequential_word_to_doc_edge_weight.pkl")

        self.parser.add_argument("--sbert_model", type=str,
                                 default="../assets/pretrained_models/"
                                         "distiluse-base-multilingual-cased-v2")
        self.parser.add_argument("--lm_model_path", type=str,
                                 default="/home/LanguageModels/bert_large_uncased")
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
