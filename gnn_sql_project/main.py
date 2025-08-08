import argparse
import json
import os
import time
from datetime import datetime

import torch
from torch_geometric.loader import NeighborLoader

import pandas as pd  # (kept since you had it; harmless if unused)
import numpy as np   # (kept since you had it; harmless if unused)
import networkx as nx  # (kept since you had it; harmless if unused)

from utils.data_loader import load_employees_db
from utils.graph_builder import create_graph
from models.gnn_model import GNN
from models.trainer import train_and_evaluate


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

    # 1) Load and prepare data
    print("\n🔍 Loading and preparing data...")
    t_data_start = time.time()
    employees, departments, dept_emp, dept_manager, titles, salaries = load_employees_db()

    print(f"\n📦 Data loaded in {time.time()-t_data_start:.2f}s")
    print(f"Employees: {employees.shape[0]:,} | Departments: {departments.shape[0]:,}")
    print(f"Relationships: {dept_emp.shape[0]:,} | Salaries: {salaries.shape[0]:,}")

    # 2) Build graph
    print("\n🛠️ Building graph...")
    t_graph_start = time.time()
    data = create_graph(employees, departments, dept_emp, dept_manager, titles, salaries)
    print(f"Graph built in {time.time()-t_graph_start:.2f}s")

    # 3) Stratified masks over EMPLOYEE nodes ONLY
    print("\n⚡ Preparing graph data...")
    num_nodes = data.num_nodes
    num_emp = int(getattr(data, 'num_employees', 0))
    assert num_emp > 0, "data.num_employees must be set by graph_builder"

    # labels for employees only; departments typically have y = -1
    y_emp = data.y[:num_emp].view(-1)
    labeled_mask = y_emp >= 0
    y_emp_labeled = y_emp[labeled_mask]

    # Stratified split per class (on labeled employees only)
    classes = torch.unique(y_emp_labeled)
    train_idx_list, val_idx_list, test_idx_list = [], [], []

    for c in classes.tolist():
        idx_c_all = torch.where(y_emp == c)[0]  # employee indices with class c
        if idx_c_all.numel() == 0:
            continue
        perm_c = idx_c_all[torch.randperm(idx_c_all.numel())]
        n_c = perm_c.numel()
        n_train = int(0.6 * n_c)
        n_val = int(0.2 * n_c)
        train_idx_list.append(perm_c[:n_train])
        val_idx_list.append(perm_c[n_train:n_train + n_val])
        test_idx_list.append(perm_c[n_train + n_val:])

    # Concatenate per-class splits
    train_idx = torch.cat(train_idx_list) if train_idx_list else torch.tensor([], dtype=torch.long)
    val_idx = torch.cat(val_idx_list) if val_idx_list else torch.tensor([], dtype=torch.long)
    test_idx = torch.cat(test_idx_list) if test_idx_list else torch.tensor([], dtype=torch.long)

    # Build boolean masks for all nodes (employees + departments)
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True

    # Debug: counts & distribution
    tr = int(data.train_mask.sum().item())
    va = int(data.val_mask.sum().item())
    te = int(data.test_mask.sum().item())
    print(f"Masks → train:{tr} val:{va} test:{te}")

    binc = torch.bincount(y_emp_labeled.to(torch.long))
    print(f"Label distribution (employees only): {binc.tolist()}")

    # 4) Model setup
    print(f"\n🚀 Starting training ({args.epochs} epochs)...")
    print(f"Batch size: {args.batch_size} | Learning rate: {args.lr}")

    input_dim = int(data.x.shape[1])
    # Prefer num_classes from graph_builder (robust to -1 labels on departments)
    num_classes = int(getattr(data, 'num_classes', 0))
    if num_classes <= 0:
        # fall back to employees portion
        num_classes = int(data.y[:num_emp].max().item()) + 1

    model = GNN(
        model_type=args.model,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=num_classes
    )

    # Count trainable params
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Model ready → {args.model} | input_dim={input_dim} | classes={num_classes}")

    # 5) Data loader over training nodes only
    train_loader = NeighborLoader(
        data,
        num_neighbors=[30, 20],
        batch_size=args.batch_size,
        input_nodes=data.train_mask,  # only sample from train employees
        shuffle=True
    )

    # 6) Train & evaluate
    test_acc = train_and_evaluate(
        model=model,
        data=data,
        train_loader=train_loader,
        epochs=args.epochs,
        lr=args.lr,
        logger=logger
    )

    # 7) Finalize logging
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
