#!/usr/bin/env python3

import psycopg2
import pandas as pd

# Force Unix socket connection (uses the 'local all all trust' rule)
conn = psycopg2.connect(
    dbname="empdb",
    user="postgres",
    host="/var/run/postgresql"
)

df = pd.read_sql("SELECT * FROM employees.employee LIMIT 10;", conn)
print(df)
