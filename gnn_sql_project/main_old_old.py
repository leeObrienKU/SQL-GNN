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
    print(f"Nodes: {len(G.nodes()):,} | Edges: {len(G.edges()):,}")

    # 3. Prepare PyTorch Geometric data
    print("\n⚡ Preparing graph data...")
    t_preprocess_start = time.time()
    
    # Feature engineering
    node_features = []
    node_mapping = {}
    for i, node in enumerate(G.nodes()):
        node_data = G.nodes[node]
        if node.startswith('emp_'):
            # Employee features
            hire_date = pd.to_datetime(node_data['hire_date'])
            tenure_days = (pd.to_datetime('today') - hire_date).days
            gender = 1 if node_data['gender'] == 'F' else 0
            node_salaries = salaries[salaries['emp_no'] == node_data['emp_no']]['salary']
            avg_salary = node_salaries.mean()/10000 if not node_salaries.empty else 0
            
            features = [
                tenure_days,    # Will be normalized
                gender,
                avg_salary,
                1               # Employee marker
            ]
        else:
            # Department features
            features = [0, 0, 0, 0]  # Department marker
            
        node_features.append(features)
        node_mapping[node] = i
    
    # Normalize features
    node_features = np.array(node_features, dtype=np.float32)
    node_features[:, 0] = (node_features[:, 0] - node_features[:, 0].min()) / \
                         (node_features[:, 0].max() - node_features[:, 0].min() + 1e-8)
    
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Create edge indices
    edge_indices = []
    for u, v in G.edges():
        edge_indices.append([node_mapping[u], node_mapping[v]])
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    
    # Create labels
    y = torch.tensor([0 if node.startswith('emp_') else 1 for node in G.nodes()], dtype=torch.long)
    
    # Create PyG data object
    data = Data(x=x, edge_index=edge_index, y=y)
    
    # Create splits
    num_nodes = len(G.nodes())
    indices = torch.randperm(num_nodes)
    data.train_mask = indices[:int(0.8*num_nodes)]
    data.val_mask = indices[int(0.8*num_nodes):int(0.9*num_nodes)]
    data.test_mask = indices[int(0.9*num_nodes):]
    
    print(f"Data processed in {time.time()-t_preprocess_start:.2f}s")

    # 4. Initialize model
    print("\n🧠 Initializing model...")
    model = GNN(
        model_type=args.model,
        input_dim=x.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=2
    )
    
    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model} | Parameters: {total_params:,} (Trainable: {trainable_params:,})")

    # 5. Training with batching
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
            "input_dim": x.shape[1],
            "hidden_dim": args.hidden_dim,
            "parameters": total_params,
            "trainable_parameters": trainable_params
        }
    )

if __name__ == "__main__":
    main()