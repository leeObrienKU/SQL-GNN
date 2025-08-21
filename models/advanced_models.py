import torch
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GATConv, SAGEConv, 
    HeteroConv, Linear, 
    TemporalConv, 
    HGTConv,
    RGCNConv
)

class HomogeneousGNN(torch.nn.Module):
    """Employee-Employee graph with similarity-based edges"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.dropout = 0.5
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Use edge weights for similarity-based connections
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv3(x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1)

class HeteroGNN(torch.nn.Module):
    """Multi-relational graph with different node and edge types"""
    def __init__(self, metadata, hidden_dim, output_dim, num_heads=4):
        super().__init__()
        
        # First layer: type-specific transforms
        self.conv1 = HeteroConv({
            ('employee', 'works_in', 'department'): GATConv((-1, -1), hidden_dim, heads=num_heads),
            ('employee', 'has_role', 'title'): GATConv((-1, -1), hidden_dim, heads=num_heads),
            ('department', 'rev_works_in', 'employee'): GATConv((-1, -1), hidden_dim, heads=num_heads),
            ('title', 'rev_has_role', 'employee'): GATConv((-1, -1), hidden_dim, heads=num_heads),
        })
        
        # Second layer
        self.conv2 = HeteroConv({
            ('employee', 'works_in', 'department'): GATConv((hidden_dim, hidden_dim), hidden_dim),
            ('employee', 'has_role', 'title'): GATConv((hidden_dim, hidden_dim), hidden_dim),
            ('department', 'rev_works_in', 'employee'): GATConv((hidden_dim, hidden_dim), hidden_dim),
            ('title', 'rev_has_role', 'employee'): GATConv((hidden_dim, hidden_dim), hidden_dim),
        })
        
        # Output layer for employee nodes
        self.lin = Linear(hidden_dim, output_dim)
        self.dropout = 0.5
        
    def forward(self, x_dict, edge_index_dict):
        # First conv layer
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) 
                 for key, x in x_dict.items()}
        
        # Second conv layer
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        # Predict on employee nodes only
        out = self.lin(x_dict['employee'])
        return F.log_softmax(out, dim=1)

class TemporalGNN(torch.nn.Module):
    """Temporal graph with time-based snapshots"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_snapshots=3):
        super().__init__()
        
        self.temporal_conv = TemporalConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=2  # look at pairs of consecutive snapshots
        )
        
        self.gnn1 = GATConv(hidden_dim, hidden_dim, heads=4)
        self.gnn2 = GATConv(hidden_dim * 4, output_dim, heads=1, concat=False)
        self.dropout = 0.5
        
    def forward(self, data_list):
        # Expect list of T graph snapshots
        # Each snapshot: (x, edge_index)
        xs = []
        for data in data_list:
            x = F.relu(self.gnn1(data.x, data.edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)
        
        # Stack features across time
        x = torch.stack(xs, dim=1)  # [N, T, F]
        
        # Apply temporal convolution
        x = self.temporal_conv(x)
        
        # Final prediction using last snapshot
        x = self.gnn2(x[:, -1], data_list[-1].edge_index)
        return F.log_softmax(x, dim=1)

class HierarchicalGNN(torch.nn.Module):
    """Two-level hierarchical graph"""
    def __init__(self, input_dim, dept_hidden_dim, emp_hidden_dim, output_dim):
        super().__init__()
        
        # Department-level convolutions
        self.dept_conv1 = GATConv(input_dim, dept_hidden_dim, heads=4)
        self.dept_conv2 = GATConv(dept_hidden_dim * 4, dept_hidden_dim, heads=1, concat=False)
        
        # Employee-level convolutions
        self.emp_conv1 = GATConv(input_dim + dept_hidden_dim, emp_hidden_dim, heads=4)
        self.emp_conv2 = GATConv(emp_hidden_dim * 4, output_dim, heads=1, concat=False)
        
        self.dropout = 0.5
        
    def forward(self, dept_data, emp_data):
        # First process department graph
        d_x = F.relu(self.dept_conv1(dept_data.x, dept_data.edge_index))
        d_x = F.dropout(d_x, p=self.dropout, training=self.training)
        d_x = self.dept_conv2(d_x, dept_data.edge_index)
        
        # Combine employee features with their department features
        dept_features = d_x[emp_data.dept_idx]  # lookup department embeddings
        e_x = torch.cat([emp_data.x, dept_features], dim=-1)
        
        # Process employee graph
        e_x = F.relu(self.emp_conv1(e_x, emp_data.edge_index))
        e_x = F.dropout(e_x, p=self.dropout, training=self.training)
        e_x = self.emp_conv2(e_x, emp_data.edge_index)
        
        return F.log_softmax(e_x, dim=1)

class KGNN(torch.nn.Module):
    """Knowledge graph with semantic relationships"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_relations):
        super().__init__()
        
        self.rgcn1 = RGCNConv(input_dim, hidden_dim, num_relations)
        self.rgcn2 = RGCNConv(hidden_dim, hidden_dim, num_relations)
        self.lin = Linear(hidden_dim, output_dim)
        self.dropout = 0.5
        
    def forward(self, data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        
        x = F.relu(self.rgcn1(x, edge_index, edge_type))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = F.relu(self.rgcn2(x, edge_index, edge_type))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.lin(x)
        return F.log_softmax(x, dim=1)
