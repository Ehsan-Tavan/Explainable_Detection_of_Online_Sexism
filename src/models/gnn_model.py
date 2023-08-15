# -*- coding: utf-8 -*-
# ========================================================
"""
    Clustering Project:
        models:
            gnn_model
"""

# ============================ Third Party libs ============================
import torch
from torch_geometric.nn import Sequential, GCNConv, GATConv
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
        # self.classifier_syntactic = torch.nn.Linear(in_features=num_feature + hidden_dim,
        #                                             out_features=hidden_dim)
        # self.classifier_sequential = torch.nn.Linear(in_features=num_feature + hidden_dim,
        #                                              out_features=hidden_dim)
        # self.classifier_semantic = torch.nn.Linear(in_features=num_feature + hidden_dim,
        #                                            out_features=hidden_dim)
        # self.conv3 = GATConv((hidden_dim // 2)*4, hidden_dim // 4, heads=4)

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

        # x = torch.cat((syntactic_x_2, sequential_x_2, semantic_x_2), dim=1)[index]

        # x_concat = torch.cat((syntactic_x, sequential_x, semantic_x), dim=1)
        # edge_index = torch.cat((syntactic_edge_index, sequential_edge_index,
        #                         semantic_edge_index), dim=1)
        #
        # x = self.conv2(x_concat, edge_index)
        # x = torch.nn.ReLU()(x)
        # x = torch.nn.functional.dropout(x, p=0.3, training=self.training)
        # x = torch.cat((x, x_concat), dim=1)[index]

        return x[index]


class Classifier(torch.nn.Module):
    def __init__(self, gnn_output_dim, lm_output_dim, num_classes):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=608,  # gnn_output_dim + lm_output_dim,
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
        self.classifier = Classifier(gnn_output_dim=hidden_dim,
                                     lm_output_dim=self.lm_model.config.hidden_size,
                                     num_classes=num_classes)
        self.attention = ScaledDotProductAttention(self.lm_model.config.hidden_size)

        self.convs = torch.nn.ModuleList([
            torch.nn.Conv2d(in_channels=1,
                            out_channels=32,
                            kernel_size=(fs, self.lm_model.config.hidden_size))
            for fs in [3, 4, 5]
        ])
        self.max_pool = torch.nn.MaxPool1d(50)

        # self.gnn_classifier = GcnClassifier(input_dim=3 * hidden_dim, num_classes=num_classes)
        # self.m = 0.6

    def forward(self, index):
        input_ids, attention_mask = self.graph.input_ids[index], self.graph.attention_mask[index]
        lm_output = self.lm_model(input_ids=input_ids,
                                  attention_mask=attention_mask)  # .last_hidden_state.unsqueeze(1)
        # last_hidden_state .pooler_output

        cnn_out = [torch.nn.ReLU()(conv(lm_output.last_hidden_state.unsqueeze(1))).squeeze(3) for
                   conv in self.convs]

        cnn_out = [torch.nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in
                   cnn_out]
        cnn_out = torch.cat(cnn_out, dim=1)

        # ------------------------ Attention Block -------------------------------
        # context, attn = self.attention(lm_output, lm_output, lm_output)
        # output = context.permute(0, 2, 1)
        # lm_output = torch.nn.functional.max_pool1d(output, output.shape[2]).squeeze(2)

        self.graph.x[index] = lm_output.pooler_output.clone().detach().requires_grad_(False)

        gnn_output = self.gnn_model(self.graph, index)

        # lm_logits = self.lm_classifier(lm_output)
        # gcn_logits = self.gnn_classifier(gnn_output)

        features = torch.cat((gnn_output, cnn_out), 1)

        pred = self.classifier(features)

        # gcn_pred = torch.nn.Softmax(dim=1)(gcn_logits)
        # lm_pred = torch.nn.Softmax(dim=1)(lm_logits)
        # pred = (gcn_pred + 1e-10) * self.m + lm_pred * (1 - self.m)
        # pred = torch.log(pred)
        return pred
