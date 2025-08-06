import os
import json
import torch
import torch.nn as nn
from math import ceil, sqrt
from pathlib import Path
import yaml
import hashlib

# --- Load experiment architecture from config
CONFIG_PATH = "config/config.yaml"
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

ARCH = config.get("experiment", {}).get("architecture", "gcn")

# --- Paths
CAT_PATH = f"output/results/graph/{ARCH}/features/categorical_info.json"
EMBED_OUTPUT = f"output/results/graph/{ARCH}/embeddings"
os.makedirs(EMBED_OUTPUT, exist_ok=True)

EMBED_PT = os.path.join(EMBED_OUTPUT, "embedding_layers.pt")
HASH_PATH = os.path.join(EMBED_OUTPUT, "catinfo.hash")

# --- Load categorical sizes
with open(CAT_PATH) as f:
    cat_info = json.load(f)

# --- Compute hash for current categorical info
cat_hash = hashlib.md5(json.dumps(cat_info, sort_keys=True).encode()).hexdigest()

# --- Check if old hash exists
if os.path.exists(HASH_PATH):
    with open(HASH_PATH) as f:
        old_hash = f.read().strip()
    if old_hash != cat_hash:
        print("⚠️ Categorical schema changed — regenerating embeddings...")
        if os.path.exists(EMBED_PT):
            os.remove(EMBED_PT)
else:
    print("📎 No previous hash found — generating embeddings...")

# --- Embedding dimension rule
def default_dim(n: int) -> int:
    return min(50, ceil(sqrt(n) * 2))  # Tunable

# --- Create embedding layers
embedding_layers = nn.ModuleDict()

for table, columns in cat_info.items():
    print(f"\n📦 Table: {table}")
    table_embeds = nn.ModuleDict()

    for col, n_uniques in columns.items():
        dim = default_dim(n_uniques)
        print(f"  🧬 {col}: {n_uniques} → dim={dim}")
        emb = nn.Embedding(num_embeddings=n_uniques + 1, embedding_dim=dim)
        table_embeds[col] = emb

    embedding_layers[table] = table_embeds

# --- Save model
torch.save(embedding_layers.state_dict(), EMBED_PT)

# --- Save architecture spec
embedding_dims = {
    table: {
        col: {
            "n_uniques": n,
            "dim": default_dim(n)
        } for col, n in cols.items()
    }
    for table, cols in cat_info.items()
}

with open(os.path.join(EMBED_OUTPUT, "embedding_dims.json"), "w") as f:
    json.dump(embedding_dims, f, indent=4)

# --- Save new hash
with open(HASH_PATH, "w") as f:
    f.write(cat_hash)

print(f"\n✅ Embedding layers + metadata saved to: {EMBED_OUTPUT}")
