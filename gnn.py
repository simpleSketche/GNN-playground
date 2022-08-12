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

################################## VISUALIZATION ##############################################
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
    plt.xtick([])
    plt.yticks([])
    plt.scatter(z[:,0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()

#################################### CREATE A SIMPLE GRAPH OBJECT ###############################

# define graph components
# x is the features of nodes,
# in this example, it is a 4x2 dimension or [4,2] array
x = torch.tensor([
    [6,4],
    [0,1],
    [5,3],
    [1,2],
])

# edge_index indicates the number of nodes for each edge and the total number of edges,
# in this example, it is a 2x8 dimension or [2,8] array
edge_index = torch.tensor([
    [0,1,0,2,1,2,2,3],
    [1,0,2,0,2,1,3,2]
])

# edge_attr indicates the edge feature for each edge,
# in this example, it is a 8x1 dimension or [8,1] array
edge_attr = torch.tensor([
    [1],
    [1],
    [4],
    [4],
    [2],
    [2],
    [5],
    [5],
])

# y is the label or so called output value,
# in this example, we are predicting node, so it is a 4x1 dimension or [4,1] array
y = torch.tensor([
    [1],
    [0],
    [1],
    [0],
])

graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
print(graph)

visualize_graph(graph)