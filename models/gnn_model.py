import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

class GNN(torch.nn.Module):
    def __init__(self, model_type='GCN', input_dim=4, hidden_dim=64, output_dim=2):
        super().__init__()
        self.model_type = model_type
        
        if model_type == 'GCN':
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, output_dim)
        elif model_type == 'GAT':
            self.conv1 = GATConv(input_dim, hidden_dim, heads=2, dropout=0.6)
            self.conv2 = GATConv(hidden_dim*2, output_dim, heads=1, concat=False, dropout=0.6)
        elif model_type == 'GraphSAGE':
            self.conv1 = SAGEConv(input_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, output_dim)
        
        self.dropout = 0.5
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        if self.model_type == 'GAT':
            x = F.elu(self.conv2(x, edge_index))
        else:
            x = self.conv2(x, edge_index)
            
        return F.log_softmax(x, dim=1)