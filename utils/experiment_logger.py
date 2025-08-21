import os
import json
import time
from datetime import datetime

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
        self.wandb_run = None

    def log_params(self, params):
        self.params = params
        print(f"\n⚙️ Experiment Parameters:")
        for k, v in params.items():
            print(f"{k:>20}: {v}")
        if self.wandb_run is not None:
            self.wandb_run.config.update(params, allow_val_change=True)

    def log_metrics(self, epoch, train_loss, val_acc, lr, val_f1=None):
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
        print(f"⏱️ Epoch {epoch:03d} | Loss: {train_loss:.4f} | Acc: {val_acc:.4f} | LR: {lr:.6f}")
        if self.wandb_run is not None:
            log_obj = {"epoch": epoch, "train/loss": train_loss, "val/acc": val_acc, "lr": lr}
            if val_f1 is not None:
                log_obj["val/f1_macro"] = val_f1
            self.wandb_run.log(log_obj)

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
        if self.wandb_run is not None:
            self.wandb_run.summary["test/acc"] = float(test_acc)
            for k, v in self.metrics.items():
                if k != "training":
                    try:
                        self.wandb_run.summary[k] = v
                    except Exception:
                        pass
