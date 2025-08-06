import json
import torch
from torch_geometric.data import HeteroData
import os
import yaml

# --- Paths
CONFIG_PATH = "config/config.yaml"
SCHEMA_JSON_PATH = "output/results/schema_graph_combined.json"  # merged output
OUTPUT_DIR = "output/results/graph/gcn"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load DB config (optional future use)
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

# --- Load schema
with open(SCHEMA_JSON_PATH) as f:
    schema = json.load(f)

# --- Initialize graph data
data = HeteroData()
metadata = {}

# --- Feature config
feature_dim = 16  # light and safe placeholder

# --- Create nodes
for table, info in schema.items():
    node_type = table
    num_nodes = info.get("row_count", 0)

    if num_nodes == 0:
        continue

    # Use zero-filled features (placeholder)
    data[node_type].x = torch.zeros((num_nodes, feature_dim))
    data[node_type].node_ids = torch.arange(num_nodes)

    metadata[node_type] = {
        "num_nodes": num_nodes,
        "primary_keys": info.get("primary_key", []),
        "columns": info.get("columns", []),
        "edges": []
    }

# --- Create edges based on foreign keys
for table, info in schema.items():
    for fk in info.get("foreign_keys", []):
        src_table = table
        dst_table = fk["references"]["table"]

        edge_type = (src_table, f"to_{dst_table}", dst_table)

        # Dummy edge index: simulate many-to-one by repeating src
        src_count = schema[src_table]["row_count"]
        dst_count = schema[dst_table]["row_count"]
        num_edges = min(src_count, dst_count, 10000)  # cap to 10k to be safe

        if num_edges == 0:
            continue

        edge_index = torch.stack([
            torch.randint(0, src_count, (num_edges,)),
            torch.randint(0, dst_count, (num_edges,))
        ], dim=0)

        data[edge_type].edge_index = edge_index

        metadata[src_table]["edges"].append({
            "to": dst_table,
            "edge_type": f"to_{dst_table}",
            "count": num_edges
        })

# --- Save outputs
torch.save(data, os.path.join(OUTPUT_DIR, "graph_data.pt"))

with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=4)

print("✅ Graph saved to:", OUTPUT_DIR)
