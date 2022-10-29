# -*- coding: utf-8 -*-
# ========================================================
"""
    Clustering Project:
        models:
            gnn_model
"""

# ============================ Third Party libs ============================
import torch
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import Sequential, GCNConv, GATConv


class GraphModel(torch.nn.Module):
    def __init__(self, num_feature, hidden_dim):
        super().__init__()
        self.conv1 = GATConv(num_feature, hidden_dim, heads=4)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim // 2, heads=4)
        # self.conv3 = GATConv((hidden_dim // 2)*4, hidden_dim // 4, heads=4)

    def forward(self, graph, index):
        x = graph.x
        edge_index, edge_attr = dropout_adj(edge_index=graph.edge_index, edge_attr=graph.edge_attr,
                                            p=0.3, training=self.training)
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.nn.ReLU()(x)
        x = torch.nn.functional.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x = torch.nn.ReLU()(x)
        x = torch.nn.functional.dropout(x, p=0.3, training=self.training)[index]
        # x = self.conv3(x, graph.edge_index, graph.edge_attr)
        # x = torch.nn.ReLU()(x)
        # x = torch.nn.functional.dropout(x, p=0.5, training=self.training)[index]
        return x


class GCNModel(torch.nn.Module):
    def __init__(self, bert_model, num_feature, hidden_dim, num_classes):
        super().__init__()
        self.lm_model = bert_model
        self.gnn_model = GraphModel(num_feature, hidden_dim)
        self.mean_pooling = torch.nn.AvgPool1d(80)
        self.classifier = torch.nn.Linear(1280, num_classes)

    def forward(self, graph, index):
        input_ids, attention_mask = graph.input_ids[index], graph.attention_mask[index]
        lm_output = self.lm_model(input_ids=input_ids,
                                  attention_mask=attention_mask).last_hidden_state
        lm_output = self.mean_pooling(lm_output.permute(0, 2, 1)).squeeze(2)

        graph.x[index] = lm_output.clone().detach().requires_grad_(True)

        gnn_output = self.gnn_model(graph, index)
        pred = torch.cat((gnn_output, lm_output), 1)
        pred = self.classifier(pred)
        return pred
