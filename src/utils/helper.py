# -*- coding: utf-8 -*-
# ========================================================
"""
    Clustering Project:
        data loader:
            data_reader
"""

# ============================ Third Party libs ============================
from math import log
from typing import List
import spacy
import torch
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced


def cal_accuracy(predictions, labels):
    pred = torch.argmax(predictions, -1).cpu().tolist()
    lab = labels.cpu().tolist()
    cor = 0
    for i in range(len(pred)):
        if pred[i] == lab[i]:
            cor += 1
    return cor / len(pred)


def calculate_tf_idf(n_docs, word_frequency, n_docs_have_word):
    idf = log(1.0 * n_docs / n_docs_have_word)
    tf_idf = word_frequency * idf
    return tf_idf


def calculate_pmi(word_pair_frequency, first_word_frequency,
                  second_word_frequency, num_window):
    pmi = log((1.0 * word_pair_frequency / num_window) /
              (1.0 * first_word_frequency * second_word_frequency / (
                      num_window * num_window)))
    return pmi


def train_step(graph, idx_loader, model, optimizer, criterion_fn, device):
    model.train()
    total_loss = 0
    total_acc = 0
    for index in idx_loader:
        (index,) = [x.to(device) for x in index]
        optimizer.zero_grad()  # Clear gradients.
        train_mask = graph.train_mask[index].type(torch.BoolTensor)

        y_pred = model(graph, index)[train_mask]

        y_true = graph.y[index][train_mask]

        # Compute the loss solely based on the training nodes.
        loss = criterion_fn(y_pred, y_true)
        # loss = torch.nn.functional.nll_loss(y_pred, y_true)

        predictions = torch.argmax(y_pred, -1).cpu().tolist()

        accuracy = accuracy_score(y_true.cpu(), predictions)

        loss.backward(retain_graph=True)  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        total_loss += loss.item()
        total_acc += accuracy
    return total_loss / len(idx_loader), total_acc / len(idx_loader)


def test_step(graph, idx_loader, model, criterion_fn, device):
    model.eval()
    y_pred = []
    y_true = []
    total_loss = 0
    for index in idx_loader:
        (index,) = [x.to(device) for x in index]
        val_mask = graph.val_mask[index].type(torch.BoolTensor)

        y_pred_ = model(graph, index)[val_mask]
        y_true_ = graph.y[index][val_mask]
        loss = criterion_fn(y_pred_, y_true_)
        predictions = torch.argmax(y_pred_, -1).cpu().tolist()
        y_pred.extend(predictions)
        y_true.extend(y_true_.cpu().tolist())

        total_loss += loss.item()
    return y_pred, y_true, total_loss / len(idx_loader)


def predict(graph, idx_loader, model, device):
    model.eval()
    y_pred = []
    with torch.no_grad():
        for index in idx_loader:
            (index,) = [x.to(device) for x in index]
            test_mask = graph.test_mask[index].type(torch.BoolTensor)
            y_pred_ = model(graph, index)[test_mask]
            predictions = torch.argmax(y_pred_, -1).cpu().tolist()
            y_pred.extend(predictions)
    return y_pred


def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h)

    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()


def create_mask(num_samples: int, doc_id_chunks: List[list]):
    mask_ids = []
    for data in doc_id_chunks:
        data_masks = []
        for idx in range(num_samples):
            if idx in data:
                data_masks.append(True)
            else:
                data_masks.append(False)
        mask_ids.append(data_masks)
    return mask_ids


def filtered_infrequent_vocabs(vocabs, vocab2frequency, min_occurrence=3):
    bad_vocabs = [vocab for vocab, count in vocab2frequency.items() if count < min_occurrence]
    return list(filter(lambda vocab: vocab not in bad_vocabs, vocabs))


def remove_stop_words_from_vocabs(vocabs):
    en = spacy.load('../assets/en_core_web_sm')
    stopwords = list(en.Defaults.stop_words)
    return list(filter(lambda vocab: vocab not in stopwords, vocabs))


def change_vocab_to_lemma(vocabs):
    en = spacy.load('../assets/en_core_web_sm')
    lemmas = set()
    for vocab in vocabs:
        doc = en(vocab)
        for token in doc:
            lemmas.add(token.lemma_.lower())
    return list(lemmas)


class F1Score(Metric):

    def __init__(self, output_transform=lambda x: x, device="cpu"):
        self.f1 = 0
        self.count = 0
        super().__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output[0].detach(), output[1].detach()

        f = f1_score(y.cpu(), y_pred.cpu(), average="macro")
        self.f1 += f
        self.count += 1

    @reinit__is_reduced
    def reset(self):
        self.f1 = 0
        self.count = 0
        super(F1Score, self).reset()

    @sync_all_reduce("_num_examples", "_num_correct:SUM")
    def compute(self):
        return self.f1 / self.count
