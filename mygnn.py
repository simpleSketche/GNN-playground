from nn_message_passing_layer import nn_message_passing_layer
from torch_geometric.nn import MessagePassing
from torch import nn
import torch

class mygnn(torch.nn.Module):
    def __init__(self, layer_num, input_dim, hidden_dim, output_dim, aggr='mean', **kwargs):
        super(mygnn, self).__init__()
        self.layer_num = layer_num

        # in case the data dimension is too small, this encoder will
        # increase the input_dim to the hidden_dim, e.g. from 2 to 30.
        self.encoder = nn.Linear(input_dim, hidden_dim)

        self.mp_layer = nn_message_passing_layer(input_dim=hidden_dim,
        hidden_dim=hidden_dim, output_dim=hidden_dim, aggr=aggr)

        # prediction
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.encoder(x)
        for i in range(self.layer_num):
            x = self.mp_layer(x, edge_index)
        node_out = self.decoder(x)
        return  node_out