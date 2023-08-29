# -*- coding: utf-8 -*-
# ========================================================
"""
    Clustering Project:
        models:
            gnn_model
"""

# ============================ Third Party libs ============================
import torch
from torch_geometric.nn import Sequential, GCNConv
from .attention import ScaledDotProductAttention


class GraphModel(torch.nn.Module):
    def __init__(self, num_feature, hidden_dim):
        super().__init__()
        self.conv1_syntactic = GCNConv(num_feature, hidden_dim)
        self.conv1_sequential = GCNConv(num_feature, hidden_dim)
        self.conv1_semantic = GCNConv(num_feature, hidden_dim)
        self.conv2_syntactic = GCNConv(hidden_dim, hidden_dim)
        self.conv2_sequential = GCNConv(hidden_dim, hidden_dim)
        self.conv2_semantic = GCNConv(hidden_dim, hidden_dim)

    def forward(self, graph, index):
        x = graph.x
        syntactic_edge_index = graph.syntactic_edge_index
        sequential_edge_index = graph.sequential_edge_index
        semantic_edge_index = graph.semantic_edge_index

        syntactic_x_1 = self.conv1_syntactic(x, syntactic_edge_index)
        syntactic_x_1 = torch.nn.ReLU()(syntactic_x_1)
        syntactic_x_1 = torch.nn.functional.dropout(syntactic_x_1, p=0.2, training=self.training)

        sequential_x_1 = self.conv1_sequential(x, sequential_edge_index)
        sequential_x_1 = torch.nn.ReLU()(sequential_x_1)
        sequential_x_1 = torch.nn.functional.dropout(sequential_x_1, p=0.2, training=self.training)

        semantic_x_1 = self.conv1_semantic(x, semantic_edge_index)
        semantic_x_1 = torch.nn.ReLU()(semantic_x_1)
        semantic_x_1 = torch.nn.functional.dropout(semantic_x_1, p=0.2, training=self.training)

        x = torch.max(syntactic_x_1, sequential_x_1)
        x = torch.max(x, semantic_x_1)

        syntactic_x_2 = self.conv2_syntactic(x, syntactic_edge_index)
        syntactic_x_2 = torch.nn.ReLU()(syntactic_x_2)
        syntactic_x_2 = torch.nn.functional.dropout(syntactic_x_2, p=0.3, training=self.training)

        sequential_x_2 = self.conv2_sequential(x, sequential_edge_index)
        sequential_x_2 = torch.nn.ReLU()(sequential_x_2)
        sequential_x_2 = torch.nn.functional.dropout(sequential_x_2, p=0.3, training=self.training)

        semantic_x_2 = self.conv2_semantic(x, semantic_edge_index)
        semantic_x_2 = torch.nn.ReLU()(semantic_x_2)
        semantic_x_2 = torch.nn.functional.dropout(semantic_x_2, p=0.3, training=self.training)

        x = torch.max(syntactic_x_2, sequential_x_2)
        x = torch.max(x, semantic_x_2)

        return x[index]


class Classifier(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=608,
                                      out_features=num_classes)

    def forward(self, features):
        features = self.linear(features)
        return features


class GCNModel(torch.nn.Module):
    def __init__(self, bert_model, graph, num_feature, hidden_dim, num_classes):
        super().__init__()
        self.lm_model = bert_model
        self.graph = graph
        self.gnn_model = GraphModel(num_feature, hidden_dim)
        self.classifier = Classifier(num_classes=num_classes)
        self.attention = ScaledDotProductAttention(self.lm_model.config.hidden_size)

        self.convs = torch.nn.ModuleList([
            torch.nn.Conv2d(in_channels=1,
                            out_channels=32,
                            kernel_size=(fs, self.lm_model.config.hidden_size))
            for fs in [3, 4, 5]
        ])
        self.max_pool = torch.nn.MaxPool1d(50)

    def forward(self, index):
        input_ids, attention_mask = self.graph.input_ids[index], self.graph.attention_mask[index]
        lm_output = self.lm_model(input_ids=input_ids,
                                  attention_mask=attention_mask)

        cnn_out = [torch.nn.ReLU()(conv(lm_output.last_hidden_state.unsqueeze(1))).squeeze(3) for
                   conv in self.convs]

        cnn_out = [torch.nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in
                   cnn_out]
        cnn_out = torch.cat(cnn_out, dim=1)

        self.graph.x[index] = lm_output.pooler_output.clone().detach().requires_grad_(False)

        gnn_output = self.gnn_model(self.graph, index)

        features = torch.cat((gnn_output, cnn_out), 1)

        pred = self.classifier(features)

        return pred
