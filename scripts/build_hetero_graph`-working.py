import os
import json
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

ARCH = "gcn"
BASE_DIR = f"output/results/graph/{ARCH}"
NODE_FEATURE_DIR = os.path.join(BASE_DIR, "node_features")
FEATURE_DIR = os.path.join(BASE_DIR, "features")
SCHEMA_PATH = "output/results/schema_graph_combined.json"
GRAPH_PATH = os.path.join(BASE_DIR, "graph_data.pt")
META_PATH = os.path.join(BASE_DIR, "graph_metadata.json")

# --- Load schema
with open(SCHEMA_PATH) as f:
    schema = json.load(f)

# --- Load node features
print("📦      Loading node features...")
data = HeteroData()
node_maps = {}

for fname in os.listdir(NODE_FEATURE_DIR):
    if fname.endswith(".pt"):
        table = fname.replace(".pt", "")
        tensor = torch.load(os.path.join(NODE_FEATURE_DIR, fname))
        data[table].x = tensor
        print(f"✅ Loaded {table} →      {tensor.shape}")

# --- Build mapping from PKs to row indices
print("\n🔎      Mapping primary keys to row indices...")
for table, info in schema.items():
    pk_cols = info.get("primary_key", [])
    if isinstance(pk_cols, str):
        pk_cols = [pk_cols]

    pk_path = os.path.join(FEATURE_DIR, f"{table}_pk_values.json")
    if not os.path.exists(pk_path):
        print(f"⚠️ No PK dump for {table}, skipping node map.")
        continue

    with open(pk_path) as f:
        pk_vals = json.load(f)

    try:
        # Handle nested list case (e.g., [["10001"], ["10002"], ...])
        if isinstance(pk_vals, list) and isinstance(pk_vals[0], list):
            pk_tuples = [tuple(map(str, row)) for row in pk_vals]
        else:
            pk_tuples = list(zip(*[pk_vals[col] for col in pk_cols]))

        node_maps[table] = {tuple(k): i for i, k in enumerate(pk_tuples)}

    except Exception as e:
        print(f"❌  Error building PK map for {table}: {e}")

# --- Build real edges from FK → PK
print("\n🔗      Building real edges...")
edge_count = 0

for src_table, info in schema.items():
    for fk in info.get("foreign_keys", []):
        fk_cols = fk["column"]
        ref_table = fk["references"]["table"]
        ref_cols = fk["references"]["column"]

        if isinstance(fk_cols, str):
            fk_cols = [fk_cols]
        if isinstance(ref_cols, str):
            ref_cols = [ref_cols]

        fk_path = os.path.join(FEATURE_DIR, f"{src_table}_fk_values.json")
        if not os.path.exists(fk_path):
            print(f"⚠️ No FK dump for {src_table}, skipping.")
            continue

        with open(fk_path) as f:
            fk_vals = json.load(f)

        try:
            fk_tuples = list(zip(*[fk_vals[col] for col in fk_cols]))
        except KeyError as e:
            print(f"❌ Missing FK column in dump for {src_table}: {e}")
            continue

        node_map = node_maps.get(ref_table)
        if not node_map:
            print(f"⚠️ No PK map for {ref_table}, skipping edge {src_table} → {ref_table}")
            continue

        edge_src = []
        edge_dst = []
        for i, fk_key in enumerate(fk_tuples):
            fk_tuple = tuple(str(v).strip() for v in fk_key)
            dst_idx = node_map.get(fk_tuple)
            if dst_idx is not None:
                edge_src.append(i)
                edge_dst.append(dst_idx)

        if edge_src:
            edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
            edge_type = (src_table, f"{src_table}_to_{ref_table}", ref_table)
            data[edge_type].edge_index = edge_index
            print(f"✅ Built edge: {src_table} → {ref_table} [{edge_index.shape[1]} edges]")
            edge_count += 1
        else:
            print(f"⚠️ No matched edges for {src_table} → {ref_table}")
            # Debug sample mismatches
            if fk_tuples:
                sample_fk = tuple(str(v).strip() for v in fk_tuples[0])
                sample_pk_keys = list(node_map.keys())[:5]
                print(f"    🔍 Sample FK: {sample_fk}")
                print(f"    🔍 Sample PK keys: {sample_pk_keys}")

# --- Save graph
torch.save(data, GRAPH_PATH)

# --- Save metadata
metadata = {
    "node_shapes": {k: list(v.x.shape) for k, v in data.items() if hasattr(v, "x")},
    "edge_types": list(data.edge_types)
}
with open(META_PATH, "w") as f:
    json.dump(metadata, f, indent=4)

print(f"\n📦  Total edge types: {edge_count}")
print(f"✅   Graph saved to: {GRAPH_PATH}")
print(f"📑   Metadata saved to: {META_PATH}")
