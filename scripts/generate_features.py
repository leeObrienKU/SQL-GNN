import os
import json
import yaml
import torch
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from tqdm import tqdm

# --- Load config
with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

schema_name = config['database']['schema']
db_name = config['database']['name']
user = config['database']['user']
batch_size = config.get('batch_size', 10000)
architecture = config.get('experiment', {}).get('architecture', 'gcn')

# --- Paths
SCHEMA_JSON_PATH = "output/results/schema_graph_combined.json"
BASE_FEATURE_DIR = f"output/results/graph/{architecture}/features"
os.makedirs(BASE_FEATURE_DIR, exist_ok=True)

categorical_sizes = {}
column_info = {}

# --- Load schema
with open(SCHEMA_JSON_PATH) as f:
    schema = json.load(f)

# --- Connect to DB
engine = create_engine(f"postgresql+psycopg2://{user}@/{db_name}")

# --- Process tables
for table_name, info in schema.items():
    row_count = info.get("row_count", 0)
    if row_count == 0:
        print(f"⏭️ Skipping empty table: {table_name}")
        continue

    print(f"\n🔄 Processing table: {table_name}")
    col_defs = info.get("columns", [])
    col_types = {list(c.keys())[0]: list(c.values())[0] for c in col_defs}
    columns = list(col_types.keys())

    numeric_cols = [col for col, dtype in col_types.items()
                    if dtype in ('integer', 'bigint', 'double precision', 'numeric', 'real', 'smallint')]
    categorical_cols = [col for col, dtype in col_types.items()
                        if dtype in ('character varying', 'character', 'text', 'USER-DEFINED', 'boolean', 'date')]

    column_info[table_name] = {
        "numeric": numeric_cols,
        "categorical": categorical_cols
    }

    offset = 0
    chunk_paths = []
    fk_values = {}
    pk_values = {}

    # --- Foreign keys
    foreign_keys = info.get("foreign_keys", [])
    fk_cols = [fk["column"] for fk in foreign_keys]

    # --- Primary keys
    pk_cols = info.get("primary_key", [])
    if isinstance(pk_cols, str):
        pk_cols = [pk_cols]

    while True:
        query = text(f"""
            SELECT * FROM {schema_name}.{table_name}
            OFFSET :offset LIMIT :limit
        """)
        with engine.connect() as conn:
            df = pd.DataFrame(conn.execute(query, {'offset': offset, 'limit': batch_size}).fetchall(), columns=columns)

        if df.empty:
            break

        df = df.dropna(axis=0, how='any')

        # --- Track FK raw values
        for fk_col in fk_cols:
            if fk_col in df.columns:
                raw_vals = df[fk_col].astype(str).tolist()
                fk_values.setdefault(fk_col, []).extend(raw_vals)

        # --- Track PK raw values
        for pk_col in pk_cols:
            if pk_col in df.columns:
                pk_values.setdefault(pk_col, []).extend(df[pk_col].astype(str).tolist())

        # --- Update categorical vocab sizes
        for cat_col in categorical_cols:
            uniques = df[cat_col].astype(str).unique()
            prev = categorical_sizes.get(table_name, {}).get(cat_col, 0)
            new = max(prev, len(uniques))
            categorical_sizes.setdefault(table_name, {})[cat_col] = new

        # --- Transform pipeline
        transformers = []
        if numeric_cols:
            transformers.append(("num", StandardScaler(), numeric_cols))
        if categorical_cols:
            transformers.append(("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_cols))

        pipe = ColumnTransformer(transformers)
        transformed = pipe.fit_transform(df)

        # --- Store tensor
        torch_tensor = torch.tensor(transformed, dtype=torch.float16)
        chunk_path = os.path.join(BASE_FEATURE_DIR, f"{table_name}_chunk_{offset}.pt")
        torch.save(torch_tensor, chunk_path)
        chunk_paths.append(chunk_path)

        print(f"  ✅ Processed rows {offset} → {offset + len(df)}")
        offset += batch_size

    # --- Save chunk list
    with open(os.path.join(BASE_FEATURE_DIR, f"{table_name}_chunks.json"), "w") as f:
        json.dump(chunk_paths, f, indent=4)

    # --- Save FK values
    if fk_values:
        fk_path = os.path.join(BASE_FEATURE_DIR, f"{table_name}_fk_values.json")
        with open(fk_path, "w") as f:
            json.dump(fk_values, f, indent=2)

    # --- Save PK values
    if pk_values:
        pk_path = os.path.join(BASE_FEATURE_DIR, f"{table_name}_pk_values.json")
        with open(pk_path, "w") as f:
            json.dump(pk_values, f, indent=2)

    print(f"✅ {table_name} done → {len(chunk_paths)} chunks")

# --- Save metadata
with open(os.path.join(BASE_FEATURE_DIR, "categorical_info.json"), "w") as f:
    json.dump(categorical_sizes, f, indent=4)

with open(os.path.join(BASE_FEATURE_DIR, "column_info.json"), "w") as f:
    json.dump(column_info, f, indent=4)

print(f"\n🎯 Feature generation completed for architecture: {architecture}")
