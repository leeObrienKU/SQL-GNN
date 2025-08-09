#!/bin/bash

echo "🚀 Starting PostgreSQL service..."
service postgresql start

echo "⬇️ Downloading empdb.dump..."
wget -O empdb.dump https://raw.githubusercontent.com/leeObrienKU/SQL-GNN/main/data/empdb.dump

echo "🛠️ Creating empdb database..."
sudo -u postgres createdb empdb

echo "📦 Restoring dump into empdb..."
sudo -u postgres pg_restore -d empdb empdb.dump

echo "🔐 Configuring pg_hba.conf for trust authentication..."
cat <<EOF > /etc/postgresql/14/main/pg_hba.conf
local   all             all                                     trust
host    all             all             127.0.0.1/32            trust
host    all             all             ::1/128                 trust
EOF

echo "🔁 Restarting PostgreSQL service..."
service postgresql restart

echo "✅ PostgreSQL setup complete."
