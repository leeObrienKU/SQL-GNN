import os
import json
import yaml
import torch
from torch import nn
from pathlib import Path
from tqdm import tqdm

# --- Load config
CONFIG_PATH = "config/config.yaml"
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

arch = config.get('experiment', {}).get('architecture', 'gcn')
FEATURE_DIR = f"output/results/graph/{arch}/features"
EMBED_PATH = f"output/results/graph/{arch}/embeddings/embedding_layers.pt"
NODE_FEATURE_DIR = f"output/results/graph/{arch}/node_features"
os.makedirs(NODE_FEATURE_DIR, exist_ok=True)

# --- Load metadata
with open(os.path.join(FEATURE_DIR, "categorical_info.json")) as f:
    categorical_info = json.load(f)
with open(os.path.join(FEATURE_DIR, "column_info.json")) as f:
    column_info = json.load(f)

# --- Load saved embedding weights
print("📥 Loading saved embedding weights from:", EMBED_PATH)
state_dict = torch.load(EMBED_PATH, map_location='cpu')

# --- Dynamically rebuild embedding layers to match saved weights
embedding_layers = nn.ModuleDict()
for key, tensor in state_dict.items():
    table, col, param = key.split(".")
    if table not in embedding_layers:
        embedding_layers[table] = nn.ModuleDict()
    if param == "weight":
        num_embeddings, emb_dim = tensor.shape
        embedding = nn.Embedding(num_embeddings, emb_dim)
        embedding_layers[table][col] = embedding

# --- Load weights
embedding_layers.load_state_dict(state_dict)
embedding_layers.eval()

# --- Assemble node features
for table, info in column_info.items():
    print(f"\n🔄 Assembling features for table: {table}")
    
    chunk_json_path = os.path.join(FEATURE_DIR, f"{table}_chunks.json")
    if not os.path.exists(chunk_json_path):
        print(f"⚠️ No chunks found for {table}, skipping.")
        continue

    with open(chunk_json_path) as f:
        chunk_paths = json.load(f)

    cat_cols = info.get("categorical", [])
    num_cols = info.get("numeric", [])
    
    embed_dict = embedding_layers[table] if table in embedding_layers else nn.ModuleDict()

    final_features = []

    for chunk_path in tqdm(chunk_paths):
        chunk = torch.load(chunk_path, map_location='cpu')

        if chunk.ndim != 2:
            print(f"❌ Invalid chunk shape: {chunk.shape}, skipping")
            continue

        # Separate numerics and categorical indices
        num_tensor = torch.empty((chunk.shape[0], 0))
        cat_tensor = torch.empty((chunk.shape[0], 0))

        if num_cols and cat_cols:
            num_tensor = chunk[:, :len(num_cols)].to(torch.float32)
            cat_tensor = chunk[:, len(num_cols):].to(torch.long)
        elif num_cols:
            num_tensor = chunk[:, :].to(torch.float32)
        elif cat_cols:
            cat_tensor = chunk[:, :].to(torch.long)

        # --- Embedding categorical columns
        cat_embeds = []
        if cat_cols:
            for i, col in enumerate(cat_cols):
                if col not in embed_dict:
                    print(f"⚠️ Column {col} not found in embeddings for {table}")
                    continue
                indices = cat_tensor[:, i].clamp(max=embed_dict[col].num_embeddings - 1)
                emb = embed_dict[col](indices)
                cat_embeds.append(emb)
        
        # --- Combine
        if cat_embeds and num_cols:
            full = torch.cat([num_tensor] + cat_embeds, dim=1)
        elif cat_embeds:
            full = torch.cat(cat_embeds, dim=1)
        elif num_cols:
            full = num_tensor
        else:
            print(f"⚠️ No usable features in {table}")
            continue

        final_features.append(full.half())  # Reduce memory

    # --- Save final feature tensor
    if final_features:
        all_rows = torch.cat(final_features, dim=0)
        save_path = os.path.join(NODE_FEATURE_DIR, f"{table}.pt")
        torch.save(all_rows, save_path)
        print(f"✅ Saved {table} → {all_rows.shape}")

    else:
        print(f"⚠️ No data assembled for {table}")
