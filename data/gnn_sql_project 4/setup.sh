#!/bin/bash

# System-level dependencies
echo "📦 Installing PostgreSQL and client tools..."
sudo apt-get update
sudo apt-get install -y postgresql postgresql-contrib postgresql-client

# Python packages
echo "🐍 Installing Python packages..."
pip install torch torch-geometric psycopg2-binary pandas networkx tabulate
