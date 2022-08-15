from torch import nn
from torch.nn.parameter import Parameter
import torch
print(torch.__version__)
import torch.nn.functional as F
import math
import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_networkx
import networkx as nx
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import sklearn.manifold
from sklearn.manifold import TSNE
import messaging_passing
import nn_message_passing_layer
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())

print(f'Dataset: {dataset}')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

print('===========================================================================================================')

graph = dataset[0]
# Gather some statistics about the graph.
print(f'Number of nodes: {graph.num_nodes}')
print(f'Number of edges: {graph.num_edges}')
print(f'Average node degree: {graph.num_edges / graph.num_nodes:.2f}')
print(f'Number of training nodes: {graph.train_mask.sum()}')
print(f'Training node label rate: {int(graph.train_mask.sum()) / graph.num_nodes:.2f}')
print(f'Has isolated nodes: {graph.has_isolated_nodes()}')
print(f'Has self-loops: {graph.has_self_loops()}')
print(f'Is undirected: {graph.is_undirected()}')


def visualize_graph(graph):
    '''
    method to visualize any given torch graph object with networkx
    '''
    # convert the graph object to networkx graph object
    vis = to_networkx(graph)
    # initialize a 8x8 matplot canvas
    plt.figure(1, figsize=(8,8))
    # documentation for networkx.draw:
    # https://networkx.org/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw.html
    nx.draw(vis, cmap=plt.get_cmap('Set3'), node_size=120, linewidths=13)
    plt.show()

def visualize_setting(h, color):
    '''
    t-SNE [1] is a tool to visualize high-dimensional data.
    It converts similarities between data points to joint probabilities and tries to minimize the Kullback-Leibler divergence between the joint probabilities of the low-dimensional embedding and the high-dimensional data.
    t-SNE has a cost function that is not convex,
    i.e. with different initializations we can get different results.
    '''
    #https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())
    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])
    plt.scatter(z[:,0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = torch_geometric.nn.GCNConv(input_dim, hidden_dim, aggr='add')
        self.conv2 = torch_geometric.nn.GCNConv(hidden_dim, output_dim, aggr='add')

    def forward(self, x, edge_index):
        print(x.shape)
        print(edge_index.shape)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# establish the model
model = GCN(input_dim=dataset.num_features,
            hidden_dim=16,
            output_dim=dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()


def train():
      model.train()
      optimizer.zero_grad()  # Clear gradients.
      out = model(graph.x, graph.edge_index)  # Perform a single forward pass.
      loss = criterion(out[graph.train_mask], graph.y[graph.train_mask])  # Compute the loss solely based on the training nodes.
      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      return loss


def test():
      model.eval()
      out = model(graph.x, graph.edge_index)
      pred = out.argmax(dim=1)  # Use the class with highest probability.
      test_correct = pred[graph.test_mask] == graph.y[graph.test_mask]  # Check against ground-truth labels.
      test_acc = int(test_correct.sum()) / int(graph.test_mask.sum())  # Derive ratio of correct predictions.
      return test_acc

for epoch in range(1, 101):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')


test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}\n')

model.eval()
out = model(graph.x, graph.edge_index)
visualize_setting(out, color=graph.y)