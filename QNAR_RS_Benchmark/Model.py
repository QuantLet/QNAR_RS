import networkx as nx
from recbole.model.abstract_recommender import SequentialRecommender, KnowledgeRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
import torch
from torch import nn, optim
from torch.nn.init import xavier_normal_, xavier_uniform_
from torch_geometric.nn import GCNConv

import logging
# https://github.com/januverma/transformers-for-sequential-recommendation/blob/main/Transformer_For_Sequential_Recommendation.ipynb - try to use this too and check results.
# check this too: https://github.com/hidasib/GRU4Rec_PyTorch_Official
class GraphLLM4Rec(SequentialRecommender):
    def __init__(self, config, dataset, torch_graph):
        super(GraphLLM4Rec, self).__init__(config, dataset)
        self.logger = logging.getLogger("GraphLLM4Rec")
        self.logger.info("GraphLLM4Rec intialized")

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        self.num_layers = config["num_layers"]
        self.dropout_prob = config["dropout_prob"]
        self.num_nodes = dataset.num(self.ITEM_ID)
        #self.batch_size = config["train_batch_size"]

        self.item_embedding = nn.Embedding(
            self.n_items, self.embedding_size, padding_idx=0
        )

        #kg_graph = dataset.kg_graph(form="coo", value_field="relation_id")
        self.torch_graph = torch_graph

        #self.lstm = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, batch_first=True, num_layers=self.num_layers, dropout=self.dropout_prob)
        self.lstm = nn.GRU(input_size=self.embedding_size, hidden_size=self.hidden_size, batch_first=True, num_layers=self.num_layers, dropout=self.dropout_prob)
        # self.lstm = nn.Transformer(d_model=self.embedding_size, num_decoder_layers=self.num_layers, num_encoder_layers=self.num_layers, batch_first=True, dropout=self.dropout_prob)

        self.gnn1 = GCNConv(self.embedding_size, self.hidden_size)
        self.gnn2 = GCNConv(256, self.hidden_size)

        # self.search_term_mlp = nn.Sequential(
        #     nn.Linear(self.embedding_size, 256),
        #     nn.ReLU(),
        #     nn.Dropout(self.dropout_prob),
        #     nn.Linear(256, 128)
        # )
        #

        # self.integration_mlp = nn.Sequential(
        #     nn.Linear(self.embedding_size + self.hidden_size, 128),
        #     nn.ReLU(),
        #     # nn.Dropout(0.2),
        #     nn.Linear(128, self.embedding_size)
        # )

        self.integration_mlp = nn.Linear(self.hidden_size, self.embedding_size)

        # parameters initialization
        self.apply(self._init_weights)

        # self.optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-4)
        # self.optimizer.zero_grad()

        if self.loss_type == "BPR":
            self.loss_function = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_function = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)

    def forward(self, search_term, item_seq, item_seq_length, edge_index, x):
        # print(item_seq, item_seq.shape)
        # print(item_seq_length, item_seq_length.shape)
        # if item_seq.dtype == torch.int64:
        #     item_seq = item_seq.to(torch.float32)
        item_seq_emb = self.item_embedding(item_seq)
        # Get the last row of the output, all columns (hidden states)
        lstm_out, _ = self.lstm(item_seq_emb)
        output = self.integration_mlp(lstm_out)

        #time_series_rep = lstm_out[:, -1, :]
        #print(time_series_rep.shape)

        # x = self.gnn1(x, edge_index)
        # x = torch.relu(x)
        # x = self.gnn2(x, edge_index)
        #gnn_rep = x.mean(dim=0).unsqueeze(0).expand(time_series_rep.shape[0], -1)
        #print(gnn_rep.shape)

        #combined_rep = torch.cat([time_series_rep, gnn_rep], dim=1)
        #print(combined_rep.shape)
        #output = self.integration_mlp(combined_rep)

        output = self.gather_indexes(output, item_seq_length - 1)

        return output

    def calculate_loss(self, interaction):
        r"""Calculate the training loss for a batch data.
        Args:
            interaction (Interaction): Interaction class of the batch.
        Returns:
            torch.Tensor: Training loss, shape: []
        """

        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]
        with torch.no_grad():
            output = self.forward(None, item_seq, item_seq_len, self.torch_graph.edge_index, self.torch_graph.x)

        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(output, test_item_emb.transpose(0, 1))

        loss = self.loss_function(logits, pos_items)

        return loss

    def predict(self, interaction):
        r"""Predict the scores between users and items.
        Args:
            interaction (Interaction): Interaction class of the batch.
        Returns:
            torch.Tensor: Predicted scores for given users and items, shape: [batch_size]
        """
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        with torch.no_grad():
            seq_output = self.forward(None, item_seq, item_seq_len, self.torch_graph.edge_index, self.torch_graph.x)

        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        with torch.no_grad():
            seq_output = self.forward(None, item_seq, item_seq_len, self.torch_graph.edge_index, self.torch_graph.x)

        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1)
        )  # [B, n_items]
        return scores