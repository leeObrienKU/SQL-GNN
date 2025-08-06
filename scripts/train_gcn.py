import os
import json
import torch
import torch.nn.functional as F
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from torch.nn import Linear, Module
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.data import HeteroData
from torch.serialization import add_safe_globals
from torch_geometric.data.storage import BaseStorage
from sqlalchemy import create_engine

# Allow torch.load to unpickle HeteroData
add_safe_globals([BaseStorage])

# === Load config ===
with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

ARCH = config["experiment"]["architecture"]
GRAPH_PATH = f"output/results/graph/{ARCH}/graph_data.pt"
ID_MAP_PATH = f"output/results/graph/{ARCH}/features/employee_pk_values.json"

print(f"📅      Loading graph from: {GRAPH_PATH}")
data: HeteroData = torch.load(GRAPH_PATH)
print("✅     Graph loaded successfully!")

# === Generate employee labels from title ===
print("🧪     Generating employee labels from title table...")

db_cfg = config["database"]
engine = create_engine(f"postgresql://{db_cfg['user']}@localhost:5432/{db_cfg['name']}")
query = """
SELECT employee_id, title
FROM employees.title
WHERE to_date = '9999-01-01'
"""
df = pd.read_sql(query, engine)

# Create class mapping for titles
title_to_class = {t: i for i, t in enumerate(df["title"].unique())}
df["label"] = df["title"].map(title_to_class)

# Load employee IDs from PK json
with open(ID_MAP_PATH) as f:
    id_list_raw = json.load(f)

# Support both list-of-lists and dict format
if isinstance(id_list_raw, list):
    id_list = [str(row[0]) for row in id_list_raw]
elif isinstance(id_list_raw, dict):
    first_key = next(iter(id_list_raw))
    id_list = [str(v) for v in id_list_raw[first_key]]
else:
    raise ValueError("Unrecognized format for employee_pk_values.json")

id_to_idx = {emp_id: i for i, emp_id in enumerate(id_list)}

# Align labels to graph nodes
labels = torch.full((len(id_list),), -1, dtype=torch.long)
for row in df.itertuples():
    idx = id_to_idx.get(str(row.employee_id))
    if idx is not None:
        labels[idx] = row.label

# Filter valid samples
mask = labels != -1
indices = torch.arange(len(labels))[mask]
labels = labels[mask]

# === Train/test/val split ===
train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

# === Define GraphSAGE model ===
class GNN(Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

model = GNN(hidden_channels=64, out_channels=len(title_to_class))
model = to_hetero(model, data.metadata(), aggr="sum")

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

print("🏋️‍♂️    Starting training...")

# === Training loop ===
target_node = "employee"
x_dict, edge_index_dict = data.x_dict, data.edge_index_dict

for epoch in range(1, 11):
    model.train()
    out_dict = model(x_dict, edge_index_dict)
    out = out_dict[target_node][mask]

    loss = loss_fn(out[train_idx], labels[train_idx])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    val_loss = loss_fn(out[val_idx], labels[val_idx])
    val_acc = (out[val_idx].argmax(dim=1) == labels[val_idx]).float().mean()

    print(f"📉 Epoch {epoch:02d} | Loss: {loss.item():.4f} | Val Acc: {val_acc:.4f}")
