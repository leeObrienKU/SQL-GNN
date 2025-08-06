import os
import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Module
from torch_geometric.nn import GCNConv, to_hetero
from torch_geometric.data import HeteroData

# === Load graph ===
ARCH = "gcn"
GRAPH_PATH = f"output/results/graph/{ARCH}/graph_data.pt"
print(f"📥   Loading graph from: {GRAPH_PATH}")

data: HeteroData = torch.load(GRAPH_PATH)
print("✅  Graph loaded successfully!")

# === Define GCN Model ===
class GCN(Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(-1, hidden_channels, add_self_loops=False)
        self.relu = ReLU()
        self.conv2 = GCNConv(hidden_channels, out_channels, add_self_loops=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        return x

# === Instantiate and transform model ===
model = GCN(hidden_channels=64, out_channels=32)

# Only node types that are destination nodes in edges:
updated_node_types = ['employee', 'department']
model = to_hetero(model, data.metadata(), aggr="sum", include_node_types=updated_node_types)

# === Dummy target + optimizer setup (adjust based on real task) ===
target_node = 'employee'
x = data[target_node].x
y = torch.randint(0, 5, (x.size(0),))  # Fake 5-class labels for demo

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()

# === Training loop (simplified) ===
for epoch in range(1, 6):
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)[target_node]
    loss = F.cross_entropy(out, y)
    loss.backward()
    optimizer.step()
    print(f"🔁 Epoch {epoch:02d} | Loss: {loss.item():.4f}")
