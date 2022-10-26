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


class GraphModel(torch.nn.Module):
    def __init__(self, num_feature, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(num_feature, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.conv3 = GCNConv(hidden_dim // 2, hidden_dim // 4)

    def forward(self, graph, index):
        x = self.conv1(graph.x, graph.edge_index)
        x = torch.nn.ReLU()(x)
        x = torch.nn.functional.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, graph.edge_index)
        x = torch.nn.ReLU()(x)
        x = torch.nn.functional.dropout(x, p=0.3, training=self.training)
        x = self.conv3(x, graph.edge_index)
        x = torch.nn.ReLU()(x)
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)[index]
        return x


class GCNModel(torch.nn.Module):
    def __init__(self, bert_model, num_feature, hidden_dim, num_classes):
        super().__init__()
        self.lm_model = bert_model
        self.gnn_model = GraphModel(num_feature, hidden_dim)
        self.mean_pooling = torch.nn.AvgPool1d(80)
        self.classifier = torch.nn.Linear(832, num_classes)

    def forward(self, graph, index):
        input_ids, attention_mask = graph.input_ids[index], graph.attention_mask[index]
        lm_output = self.lm_model(input_ids).last_hidden_state
        lm_output = self.mean_pooling(lm_output.permute(0, 2, 1)).squeeze(2)

        graph.x[index] = lm_output.clone().detach().requires_grad_(True)

        gnn_output = self.gnn_model(graph, index)
        pred = torch.cat((gnn_output, lm_output), 1)
        pred = self.classifier(pred)
        return pred

# class Encoder(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super().__init__()
#         self.conv1 = GCNConv(in_channels=in_channels, out_channels=hidden_channels)
#         self.conv_mu = GCNConv(in_channels=hidden_channels, out_channels=out_channels)
#         self.conv_logstd = GCNConv(in_channels=hidden_channels, out_channels=out_channels)
#
#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index).relu()
#         return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
#
#
# class Discriminator(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super().__init__()
#         self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
#         self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels)
#         self.lin3 = torch.nn.Linear(hidden_channels, out_channels)
#
#     def forward(self, x):
#         x = self.lin1(x).relu()
#         x = self.lin2(x).relu()
#         return self.lin3(x)
#
#
# class AutoEncoder(LightningModule):
#     def __init__(self, num_feature: int, hidden_channels: int, out_channels: int):
#         super().__init__()
#         self.encoder = Encoder(in_channels=num_feature, hidden_channels=hidden_channels,
#                                out_channels=out_channels)
#         self.discriminator = Discriminator(in_channels=hidden_channels,
#                                            hidden_channels=2 * hidden_channels,
#                                            out_channels=hidden_channels)
#         self.model = ARGVA(self.encoder, self.discriminator)
#
#         self.save_hyperparameters()
#
#     def forward(self, x, edge_index):
#         return self.model.encode(x, edge_index)
#
#     def training_step(self, data, _):
#         y_hat = self(data.x, data.edge_index)[:data.batch_size]
#
#     def configure_optimizers(self):
#         encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=0.005)
#         discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(),
#                                                    lr=0.001)
#
#         return encoder_optimizer, discriminator_optimizer
