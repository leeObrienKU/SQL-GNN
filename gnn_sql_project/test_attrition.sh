#!/bin/bash
set -euo pipefail

# Move to project root
cd /content/gnn_sql_project

# GAT-friendly defaults for the attrition (leaver) task
python main.py \
  --task attrition \
  --cutoff 2000-01-01 \
  --model GraphSAGE \
  --epochs 100 \
  --batch_size 512 \
  --hidden_dim 128 \
  --lr 0.0015
