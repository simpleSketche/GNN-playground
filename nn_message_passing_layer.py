'''
powerful approach:
let the model to train and learn to choose better message passing
mechanism using neuarl network,
needs a large dimension data
'''
from torch_geometric.nn import MessagePassing
from torch import nn
import torch

class nn_message_passing_layer(MessagePassing):
    def  __init__(self, input_dim, hidden_dim, output_dim, aggr='mean'):
        super(nn_message_passing_layer, self).__init__()
        self.aggr = aggr
        self.messageNN = nn.Linear(input_dim * 2, hidden_dim)
        self.updateNN = nn.Linear(input_dim + hidden_dim, output_dim)
    
    def forward(self, x, edge_index):
        return self.propgate(edge_index, x=x, messageNN=self.messageNN, updateNN=self.updateNN)
    
    def message(self, x_i, x_j, messageNN):
        return messageNN(torch.cat((x_i, x_j), dim=-1))
    
    def update(self, aggr_out, x, updateNN):
        return updateNN(torch.cat((x, aggr_out), dim=-1))