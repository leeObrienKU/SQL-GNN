#!/bin/bash

# Ensure subversion is installed
echo "📦 Installing subversion..."
apt-get update -qq
apt-get install -y subversion

# Download only the gnn_sql_project folder from GitHub repo
echo "⬇️ Downloading gnn_sql_project folder from GitHub..."
svn export https://github.com/leeObrienKU/SQL-GNN/trunk/gnn_sql_project /content/gnn_sql_project

echo "✅ Download complete: /content/gnn_sql_project"
