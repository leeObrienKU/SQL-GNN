import torch
import psycopg2
from packaging import version
import warnings

def check_cuda_memory():
    if torch.cuda.is_available():
        print(f"CUDA Memory Allocation:")
        print(f"- Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        print(f"- Cached: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
    else:
        print("CUDA not available")

def verify_db_connection(dbname="empdb", user="postgres", host="/var/run/postgresql"):
    try:
        conn = psycopg2.connect(dbname=dbname, user=user, host=host)
        conn.close()
        return True
    except Exception as e:
        warnings.warn(f"Database connection failed: {str(e)}")
        return False
