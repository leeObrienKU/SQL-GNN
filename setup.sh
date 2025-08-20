#!/bin/bash
set -e

echo "📦 Installing system packages..."
apt-get update -qq
apt-get install -y postgresql postgresql-contrib postgresql-client

echo "🐍 Installing Python packages..."
pip install --quiet torch torch-geometric psycopg2-binary pandas networkx tabulate

# Install PyG optional GPU acceleration libs
pip install --quiet pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-$(python3 -c "import torch; print(torch.__version__)").html

echo "✅ Environment setup complete."
