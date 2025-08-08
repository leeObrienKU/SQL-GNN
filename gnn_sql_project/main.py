class ExperimentLogger:
    def __init__(self):
        self.start_time = time.time()
        self.log_dir = "experiment_logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"{self.log_dir}/experiment_{self.experiment_id}.json"
        self.params = {}
        self.metrics = {
            "training": [],
            "validation": [],
            "test": None
        }

    def log_params(self, params):
        self.params = params
        print(f"\n⚙️ Experiment Parameters:")
        for k, v in params.items():
            print(f"{k:>20}: {v}")

    def log_metrics(self, epoch, train_loss, val_acc, lr):
        entry = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_acc": float(val_acc),
            "lr": float(lr),
            "timestamp": time.time()
        }
        self.metrics["training"].append(entry)
        print(f"⏱️ Epoch {epoch:03d} | Loss: {train_loss:.4f} | Acc: {val_acc:.4f} | LR: {lr:.6f}")

    def finalize(self, test_acc, model_summary):
        runtime = time.time() - self.start_time
        self.metrics["test"] = float(test_acc)
        self.metrics["runtime_seconds"] = runtime
        self.metrics["model_summary"] = model_summary
        with open(self.log_file, 'w') as f:
            json.dump({
                "params": self.params,
                "metrics": self.metrics
            }, f, indent=2)
        print(f"\n✅ Experiment complete in {runtime:.2f} seconds")
        print(f"📊 Results saved to {self.log_file}")


import argparse
import pandas as pd
import torch
import numpy as np
import networkx as nx
import time
from datetime import datetime
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, NeighborLoader
from utils.data_loader import load_employees_db
from utils.graph_builder import create_graph
from models.gnn_model import GNN
from models.trainer import train_and_evaluate
import json
import os

        
def main():
    # Initialize logging
    logger = ExperimentLogger()
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='GNN Employee Database Training')
    parser.add_argument('--model', type=str, default='GCN', 
                      choices=['GCN', 'GAT', 'GraphSAGE'],
                      help='Type of GNN model')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1024,
                      help='Batch size for neighbor sampling')
    parser.add_argument('--hidden_dim', type=int, default=64,
                      help='Hidden dimension size')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate')
    args = parser.parse_args()
    
    # Log parameters
    logger.log_params(vars(args))
    
    # 1. Load and prepare data
    print("\n🔍 Loading and preparing data...")
    t_data_start = time.time()
    employees, departments, dept_emp, dept_manager, titles, salaries = load_employees_db()
    
    # Debug prints
    print(f"\n📦 Data loaded in {time.time()-t_data_start:.2f}s")
    print(f"Employees: {employees.shape[0]:,} | Departments: {departments.shape[0]:,}")
    print(f"Relationships: {dept_emp.shape[0]:,} | Salaries: {salaries.shape[0]:,}")

    # 2. Build graph
    print("\n🛠️ Building graph...")
    t_graph_start = time.time()
    data = create_graph(employees, departments, dept_emp, dept_manager, titles, salaries)
    print(f"Graph built in {time.time()-t_graph_start:.2f}s")

    # 3. Prepare PyTorch Geometric data
    print("\n⚡ Preparing graph data...")
    t_preprocess_start = time.time()
    
    # Feature engineering
            
            
    
    
    
    # Create edge indices
    edge_indices = []
    
    # Create labels
    
    # Create PyG data object
    # Create train/val/test masks based on node count from PyG Data
    num_nodes = data.num_nodes
    perm = torch.randperm(num_nodes)
    n_train = int(0.6 * num_nodes)
    n_val = int(0.2 * num_nodes)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True

    # 5. Training with batching
    # --- Sanity checks: label distribution & mask sizes ---
    try:
        if hasattr(data, 'y') and data.y is not None and data.y.numel() > 0:
            binc = torch.bincount(data.y.flatten().to(torch.long))
        # Warn if labels are constant (only one class)
        try:
            unique_classes = int((binc > 0).sum().item())
            if unique_classes <= 1:
                print("⚠️  Warning: labels are constant — training will trivially hit 100% accuracy.")
        except Exception as _e:
            pass
            print(f"Label distribution: {binc.tolist()}")
        else:
            print("Label distribution: <no labels>")
    except Exception as e:
        print(f"Label distribution check failed: {e}")
    try:
        tr = int(data.train_mask.sum().item()) if hasattr(data, 'train_mask') else 0
        va = int(data.val_mask.sum().item()) if hasattr(data, 'val_mask') else 0
        te = int(data.test_mask.sum().item()) if hasattr(data, 'test_mask') else 0
        print(f"Masks → train:{tr} val:{va} test:{te}")
    except Exception as e:
        print(f"Mask size check failed: {e}")

    print(f"\n🚀 Starting training ({args.epochs} epochs)...")
    print(f"Batch size: {args.batch_size} | Learning rate: {args.lr}")
    
    # Create data loaders
    train_loader = NeighborLoader(
        data,
        num_neighbors=[30, 20],
        batch_size=args.batch_size,
        input_nodes=data.train_mask,
        shuffle=True
    )
    
    # Train and evaluate

    # --- Model init (inserted) ---
    input_dim = int(data.x.shape[1])
    if hasattr(data, 'y') and data.y is not None and data.y.numel() > 0:
        num_classes = int(data.y.max().item()) + 1
    else:
        num_classes = 2
    model = GNN(model_type=args.model, input_dim=input_dim, hidden_dim=args.hidden_dim, output_dim=num_classes)
    # --- Count trainable parameters ---
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    print(f"Model ready → {args.model} | input_dim={input_dim} | classes={num_classes}")

    test_acc = train_and_evaluate(
        model=model,
        data=data,
        train_loader=train_loader,
        epochs=args.epochs,
        lr=args.lr,
        logger=logger
    )
    
    # Finalize logging
    logger.finalize(
        test_acc=test_acc,
        model_summary={
            "type": args.model,
            "input_dim": data.x.shape[1],
            "hidden_dim": args.hidden_dim,
            "parameters": total_params,
            "trainable_parameters": total_params
        }
    )


    print("\n📊 Final Training Summary")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Params: {sum(p.numel() for p in model.parameters())} total parameters")

if __name__ == "__main__":
    main()