import os
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np

class ExperimentLogger:
    def __init__(self):
        self.start_time = time.time()
        self.base_dir = "experiment_logs"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create experiment directory with timestamp
        self.experiment_id = None  # Will be set when model type is known
        self.log_dir = None  # Will be set when model type is known
        os.makedirs(self.base_dir, exist_ok=True)
        
        self.params = {}
        self.metrics = {
            "training": [],
            "validation": [],
            "test": None,
            "confusion_matrix": None
        }
        self.wandb_run = None
        self.performance_summary = None

    def _setup_directories(self, model_name, graph_type):
        """Setup directory structure once model type is known"""
        self.experiment_id = f"{model_name}_{graph_type}_{self.timestamp}"
        self.log_dir = os.path.join(self.base_dir, self.experiment_id)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create subdirectories
        self.plot_dir = os.path.join(self.log_dir, "plots")
        self.metrics_dir = os.path.join(self.log_dir, "metrics")
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

    def log_params(self, params):
        """Log experiment parameters"""
        self.params = params
        
        # Setup directories now that we know the model type
        self._setup_directories(params['model'], params['graph_type'])
        
        # Save parameters to file
        params_file = os.path.join(self.log_dir, "parameters.json")
        with open(params_file, 'w') as f:
            json.dump(params, f, indent=2)
        
        # Print parameters
        print(f"\n⚙️ Experiment Parameters:")
        for k, v in params.items():
            print(f"{k:>20}: {v}")
        
        if self.wandb_run is not None:
            self.wandb_run.config.update(params, allow_val_change=True)

    def log_metrics(self, epoch, train_loss, val_acc, lr, val_f1=None):
        """Log training metrics for each epoch"""
        entry = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_acc": float(val_acc),
            "lr": float(lr),
            "timestamp": time.time()
        }
        if val_f1 is not None:
            entry["val_f1_macro"] = float(val_f1)
        
        self.metrics["training"].append(entry)
        
        # Save current training metrics
        metrics_file = os.path.join(self.metrics_dir, "training_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics["training"], f, indent=2)
        
        print(f"⏱️ Epoch {epoch:03d} | Loss: {train_loss:.4f} | Acc: {val_acc:.4f} | LR: {lr:.6f}")
        
        if self.wandb_run is not None:
            log_obj = {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/acc": val_acc,
                "lr": lr
            }
            if val_f1 is not None:
                log_obj["val/f1_macro"] = val_f1
            self.wandb_run.log(log_obj)

    def log_confusion_matrix(self, cm, class_names):
        """Log confusion matrix"""
        self.metrics["confusion_matrix"] = {
            "matrix": cm.tolist(),
            "class_names": class_names
        }
        
        # Save confusion matrix
        cm_file = os.path.join(self.metrics_dir, "confusion_matrix.json")
        with open(cm_file, 'w') as f:
            json.dump(self.metrics["confusion_matrix"], f, indent=2)

    def save_plot(self, fig, name):
        """Save a matplotlib figure with consistent naming"""
        plot_path = os.path.join(self.plot_dir, f"{name}_{self.timestamp}.png")
        fig.savefig(plot_path)
        return plot_path

    def finalize(self, test_acc, model_summary):
        """Finalize the experiment and save all results"""
        runtime = time.time() - self.start_time
        self.metrics["test"] = float(test_acc)
        self.metrics["runtime_seconds"] = runtime
        self.metrics["model_summary"] = model_summary
        
        # Create performance summary
        self.performance_summary = {
            "model_type": model_summary["type"],
            "graph_type": model_summary["graph_type"],
            "timestamp": self.timestamp,
            "test_accuracy": float(test_acc),
            "runtime_seconds": runtime,
            "num_parameters": model_summary["parameters"],
            "best_val_accuracy": max(e["val_acc"] for e in self.metrics["training"]),
            "final_train_loss": self.metrics["training"][-1]["train_loss"],
            "epochs": len(self.metrics["training"])
        }
        
        # Save all metrics
        metrics_file = os.path.join(self.log_dir, "experiment_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump({
                "params": self.params,
                "metrics": self.metrics
            }, f, indent=2)
        
        # Save performance summary
        summary_file = os.path.join(self.metrics_dir, f"performance_{self.timestamp}.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Performance Summary for {self.experiment_id}\n")
            f.write("=" * 80 + "\n\n")
            for k, v in self.performance_summary.items():
                f.write(f"{k:.<40} {v}\n")
            
            # Add confusion matrix if available
            if self.metrics["confusion_matrix"] is not None:
                f.write("\nConfusion Matrix:\n")
                cm = np.array(self.metrics["confusion_matrix"]["matrix"])
                class_names = self.metrics["confusion_matrix"]["class_names"]
                f.write("\nClass names: " + ", ".join(class_names) + "\n")
                f.write("\n" + str(cm) + "\n")
                
                # Calculate additional metrics
                if cm.shape == (2, 2):  # Binary classification
                    tn, fp, fn, tp = cm.ravel()
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    f.write("\nDetailed Metrics:\n")
                    f.write(f"Precision: {precision:.4f}\n")
                    f.write(f"Recall: {recall:.4f}\n")
                    f.write(f"F1 Score: {f1:.4f}\n")
        
        # Update master results file
        master_file = os.path.join(self.base_dir, "all_experiments.csv")
        df_row = pd.DataFrame([self.performance_summary])
        if os.path.exists(master_file):
            df = pd.read_csv(master_file)
            df = pd.concat([df, df_row], ignore_index=True)
        else:
            df = df_row
        df.to_csv(master_file, index=False)
        
        print(f"\n✅ Experiment complete in {runtime:.2f} seconds")
        print(f"📊 Results saved to {self.log_dir}")
        print(f"📈 Performance summary saved to {summary_file}")
        
        if self.wandb_run is not None:
            self.wandb_run.summary["test/acc"] = float(test_acc)
            for k, v in self.metrics.items():
                if k != "training":
                    try:
                        self.wandb_run.summary[k] = v
                    except Exception:
                        pass