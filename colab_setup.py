import os
import subprocess
from pathlib import Path

def setup_colab_environment():
    """Setup the Colab environment with required dependencies and file structure"""
    print("Setting up Colab environment...")
    
    # Install dependencies
    print("\nInstalling dependencies...")
    packages = [
        "torch",
        "torch_geometric",
        "pandas",
        "numpy",
        "tqdm",
        "wandb"
    ]
    for package in packages:
        subprocess.run(["pip", "install", package])
    
    # Create directory structure
    print("\nCreating directory structure...")
    directories = [
        "data",
        "experiment_logs",
        "models",
        "utils",
        "wandb"
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Verify installation
    print("\nVerifying installation...")
    import torch
    import torch_geometric
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    import wandb
    
    print("\nEnvironment setup complete!")
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch Geometric version: {torch_geometric.__version__}")
    
    # Verify file structure
    print("\nCurrent directory structure:")
    subprocess.run(["ls", "-R"])

if __name__ == "__main__":
    setup_colab_environment()
