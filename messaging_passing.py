from torch_geometric.nn import MessagePassing

class self_designed_MessagePassingLayer(MessagePassing):
    def __init__(self):
        super(self_designed_MessagePassingLayer, self).__init__()
        # the line below defines the method of aggregation
        # add > mean > max
        self.aggr = aggr

    def forward(self, x, edge_index):
        '''
        forward method feeds data to the model,
        once this method is called, it will automatically
        trigger message method first and then update method.

        x: the central node which receives message
        edge_index: the indexes of the connecting edges of the central node
        '''
        return self.propagate(edge_index,x=x)

    def message(self, x_i, x_j):
        '''
        message method literally creates messages to propagate between nodes,
        therefore, need to define here what kind of message to pass.

        x_i: the central node
        x_j: the current neighboring node

        sample message below: half of the central node value is added to twice of the current neighboring node value
        '''
        return 0.5 * x_i + 2 * x_j

    def update(self, aggr_out, x):
        return None