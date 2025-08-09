#!/bin/bash
cd /content/gnn_sql_project

python main.py \
  --model GCN \
  --epochs 100 \
  --batch_size 2048 \
  --hidden_dim 256 \
  --lr 0.002 \
  --task attrition \
  --cutoff 2000-01-01 
  #--current_edges_only comment out for denser
