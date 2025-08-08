#!/bin/bash

# Navigate to the project directory (adjust if needed)
cd /content/gnn_sql_project

# Run the main script with sample parameters
python main.py \
  --model GCN \
  --epochs 100 \
  --batch_size 2048 \
  --hidden_dim 128 \
  --lr 0.0005
