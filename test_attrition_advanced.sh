#!/bin/bash

# Base parameters for all experiments
BASE_PARAMS="--epochs 100 \
  --batch_size 2048 \
  --hidden_dim 256 \
  --lr 0.002 \
  --lr_decay_type exponential \
  --lr_decay_gamma 0.98 \
  --task attrition \
  --cutoff 2002-12-31 \
  --wandb --wandb_project sql_to_gnn"

# 1. Employee-Employee Graph (Homogeneous)
echo "🤝 Running Employee-Employee Graph Experiment..."
python main.py \
  --model GCN \
  --graph_type homogeneous \
  --similarity_threshold 0.7 \
  $BASE_PARAMS \
  --experiment_name "homogeneous_emp_emp"

# 2. Multi-relational Graph (Heterogeneous)
echo "🌐 Running Heterogeneous Graph Experiment..."
python main.py \
  --model HeteroGNN \
  --graph_type heterogeneous \
  --num_heads 4 \
  $BASE_PARAMS \
  --experiment_name "heterogeneous_multi_rel"

# 3. Temporal Dynamic Graph
echo "⏰ Running Temporal Graph Experiment..."
python main.py \
  --model TemporalGNN \
  --graph_type temporal \
  --time_windows "2000-01-01,2001-01-01,2002-01-01" \
  --prediction_horizon "6M" \
  $BASE_PARAMS \
  --experiment_name "temporal_dynamic"

# 4. Hierarchical Graph
echo "📊 Running Hierarchical Graph Experiment..."
python main.py \
  --model HierarchicalGNN \
  --graph_type hierarchical \
  --dept_hidden_dim 128 \
  --emp_hidden_dim 256 \
  $BASE_PARAMS \
  --experiment_name "hierarchical_org"

# 5. Knowledge Graph
echo "🧠 Running Knowledge Graph Experiment..."
python main.py \
  --model KGNN \
  --graph_type knowledge \
  --num_relations 5 \
  --embedding_dim 128 \
  $BASE_PARAMS \
  --experiment_name "knowledge_graph"

# Evaluation script to compare all models
echo "📈 Running Model Comparison..."
python compare_models.py \
  --results_dir "experiment_logs" \
  --output_file "model_comparison.pdf" \
  --metrics "accuracy,auc,precision,recall,f1"
