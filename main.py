import argparse
import json
import os
import time
from datetime import datetime

import torch
from torch_geometric.loader import NeighborLoader

from utils.plots import plot_training_curves, plot_confusion_matrix
from utils.experiment_logger import ExperimentLogger
import numpy as np
import pandas as pd
import networkx as nx

from utils.data_loader import load_employees_db
from utils.graph_builder import create_graph
from utils.advanced_graph_builder import AdvancedGraphBuilder
from models.gnn_model import GNN
from models.advanced_models import (
    HomogeneousGNN, HeteroGNN, TemporalGNN,
    HierarchicalGNN, KGNN
)
from models.trainer import train_and_evaluate

# Weights & Biases (optional)
try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    wandb = None
    _WANDB_AVAILABLE = False

def main():
    parser = argparse.ArgumentParser(description='GNN Employee Database Training')
    
    # Model selection
    parser.add_argument('--model', type=str, default='GCN',
                        choices=['GCN', 'GAT', 'GraphSAGE', 
                                'HomogeneousGNN', 'HeteroGNN', 'TemporalGNN',
                                'HierarchicalGNN', 'KGNN'],
                        help='Type of GNN model')
    
    # Graph structure
    parser.add_argument('--graph_type', type=str, default='bipartite',
                        choices=['bipartite', 'homogeneous', 'heterogeneous', 
                                'temporal', 'hierarchical', 'knowledge'],
                        help='Type of graph structure to use')
    
    # Advanced model parameters
    parser.add_argument('--similarity_threshold', type=float, default=0.7,
                        help='Similarity threshold for homogeneous graph')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads for GAT/HeteroGNN')
    parser.add_argument('--num_relations', type=int, default=5,
                        help='Number of relation types for knowledge graph')
    parser.add_argument('--dept_hidden_dim', type=int, default=128,
                        help='Hidden dimension for department-level GNN')
    parser.add_argument('--emp_hidden_dim', type=int, default=256,
                        help='Hidden dimension for employee-level GNN')
    parser.add_argument('--time_windows', type=str, default="2000-01-01,2001-01-01,2002-01-01",
                        help='Comma-separated list of dates for temporal snapshots')
    parser.add_argument('--prediction_horizon', type=str, default="6M",
                        help='Prediction horizon for temporal model (e.g., 6M for 6 months)')
    
    # Experiment tracking
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for this experiment run')
    
    # Original parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for neighbor sampling')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--lr_decay_type', type=str, default='exponential',
                        choices=['none','exponential','step'],
                        help='Learning rate decay scheduler type')
    parser.add_argument('--lr_decay_gamma', type=float, default=0.95,
                        help='LR decay factor (gamma)')
    parser.add_argument('--lr_decay_step_size', type=int, default=20,
                        help='Step size for StepLR (epochs)')
    parser.add_argument('--task', type=str, default='attrition',
                        choices=['dept', 'attrition'],
                        help='Prediction task')
    parser.add_argument('--cutoff', type=str, default="2002-12-31",
                        help='Cutoff date YYYY-MM-DD for attrition labeling')
    parser.add_argument('--current_edges_only', action='store_true',
                        help='Keep only current edges')
    parser.add_argument('--pos_threshold', type=float, default=0.5,
                        help='Positive class threshold')
    parser.add_argument('--auto_threshold', action='store_true',
                        help='Auto-select threshold')
    parser.add_argument('--threshold_mode', type=str, default=None,
                        choices=['fixed', 'max_f1', 'target_precision', 'target_recall'],
                        help='Threshold selection mode')
    parser.add_argument('--target_precision', type=float, default=None,
                        help='Target precision')
    parser.add_argument('--target_recall', type=float, default=None,
                        help='Target recall')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable W&B logging')
    parser.add_argument('--wandb_project', type=str, default='sql_to_gnn',
                        help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='W&B entity')
    parser.add_argument('--wandb_api_key', type=str, default=None,
                        help='W&B API key')
    
    args = parser.parse_args()
    
    # Initialize logging
    logger = ExperimentLogger()
    if args.experiment_name:
        logger.experiment_id = args.experiment_name
    
    # W&B setup
    if args.wandb and _WANDB_AVAILABLE:
        api_key = args.wandb_api_key or os.environ.get('WANDB_API_KEY')
        if api_key:
            os.environ['WANDB_API_KEY'] = api_key
        logger.wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config={},
            name=f"{args.model}-{args.graph_type}-{logger.experiment_id}"
        )
    
    # Log parameters
    logger.log_params(vars(args))
    
    # Load data
    print("\n🔍 Loading and preparing data...")
    employees, departments, dept_emp, dept_manager, titles, salaries = load_employees_db()
    
    # Build graph based on type
    print(f"\n🛠️ Building {args.graph_type} graph...")
    builder = AdvancedGraphBuilder()
    
    if args.graph_type == 'bipartite':
        data = create_graph(
            employees=employees,
            departments=departments,
            dept_emp=dept_emp,
            dept_manager=dept_manager,
            titles=titles,
            salaries=salaries,
            task=args.task,
            cutoff_date=args.cutoff,
            use_all_history_edges=not args.current_edges_only
        )
    elif args.graph_type == 'homogeneous':
        data = builder.create_employee_graph(
            employees, departments, dept_emp, titles, salaries,
            cutoff_date=args.cutoff,
            similarity_threshold=args.similarity_threshold
        )
    elif args.graph_type == 'heterogeneous':
        data = builder.create_heterogeneous_graph(
            employees, departments, dept_emp, titles, salaries,
            cutoff_date=args.cutoff
        )
    elif args.graph_type == 'temporal':
        time_windows = args.time_windows.split(',')
        data = builder.create_temporal_graph(
            employees, departments, dept_emp, titles, salaries,
            time_windows=time_windows,
            prediction_horizon=args.prediction_horizon
        )
    elif args.graph_type == 'hierarchical':
        dept_graph, emp_graphs = builder.create_hierarchical_graph(
            employees, departments, dept_emp, dept_manager,
            titles, salaries, cutoff_date=args.cutoff
        )
        data = (dept_graph, emp_graphs)
    elif args.graph_type == 'knowledge':
        data = builder.create_knowledge_graph(
            employees, departments, dept_emp, titles, salaries,
            cutoff_date=args.cutoff
        )
    
    # Create appropriate model
    input_dim = data.x.shape[1] if hasattr(data, 'x') else data[0].x.shape[1]
    num_classes = getattr(data, 'num_classes', 2)
    
    if args.model in ['GCN', 'GAT', 'GraphSAGE']:
        model = GNN(
            model_type=args.model,
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=num_classes
        )
    elif args.model == 'HomogeneousGNN':
        model = HomogeneousGNN(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=num_classes
        )
    elif args.model == 'HeteroGNN':
        model = HeteroGNN(
            metadata=data.metadata(),
            hidden_dim=args.hidden_dim,
            output_dim=num_classes,
            num_heads=args.num_heads
        )
    elif args.model == 'TemporalGNN':
        model = TemporalGNN(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=num_classes,
            num_snapshots=len(args.time_windows.split(','))
        )
    elif args.model == 'HierarchicalGNN':
        model = HierarchicalGNN(
            input_dim=input_dim,
            dept_hidden_dim=args.dept_hidden_dim,
            emp_hidden_dim=args.emp_hidden_dim,
            output_dim=num_classes
        )
    elif args.model == 'KGNN':
        model = KGNN(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=num_classes,
            num_relations=args.num_relations
        )
    
    # Training setup
    train_loader = NeighborLoader(
        data,
        num_neighbors=[30, 20],
        batch_size=args.batch_size,
        input_nodes=data.train_mask,
        shuffle=True
    )
    
    # Train and evaluate
    threshold_mode = args.threshold_mode or ('max_f1' if args.auto_threshold else 'fixed')
    test_acc = train_and_evaluate(
        model=model,
        data=data,
        train_loader=train_loader,
        epochs=args.epochs,
        lr=args.lr,
        logger=logger,
        pos_threshold=args.pos_threshold,
        auto_threshold=args.auto_threshold,
        threshold_mode=threshold_mode,
        target_precision=args.target_precision,
        target_recall=args.target_recall,
        lr_decay_type=args.lr_decay_type,
        lr_decay_gamma=args.lr_decay_gamma,
        lr_decay_step_size=args.lr_decay_step_size
    )
    
    # Save plots
    out_dir = logger.log_dir
    plot_training_curves(logger.metrics["training"], out_dir)
    cm = np.array(logger.metrics.get("confusion_matrix", [[0, 0], [0, 0]]))
    class_names = ["Stay", "Leave"] if args.task == "attrition" and cm.shape == (2, 2) \
                  else [str(i) for i in range(cm.shape[0])]
    plot_confusion_matrix(cm, class_names, os.path.join(out_dir, "confusion_matrix.png"))
    
    # Log to W&B
    if logger.wandb_run is not None:
        try:
            logger.wandb_run.log({
                "plots/training_loss": wandb.Image(os.path.join(out_dir, "training_loss.png")),
                "plots/val_accuracy": wandb.Image(os.path.join(out_dir, "val_accuracy.png")),
                "plots/confusion_matrix": wandb.Image(os.path.join(out_dir, "confusion_matrix.png")),
                "confusion_matrix/raw": cm
            })
        except Exception:
            pass
    
    # Finalize
    logger.finalize(
        test_acc=test_acc,
        model_summary={
            "type": args.model,
            "graph_type": args.graph_type,
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "parameters": sum(p.numel() for p in model.parameters())
        }
    )
    
    if logger.wandb_run is not None:
        try:
            logger.wandb_run.finish()
        except Exception:
            pass
    
    print("\n📊 Final Training Summary")
    print(f"Model: {args.model} on {args.graph_type} graph")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())} total")

if __name__ == "__main__":
    main()