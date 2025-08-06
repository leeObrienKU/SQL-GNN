import yaml
import sqlalchemy
import pandas as pd

# Load database info from config
with open("/content/config/config.yaml") as f:
    config = yaml.safe_load(f)

db_name = config['database']['name']
db_user = config['database']['user']
db_schema = config['database']['schema']

engine = sqlalchemy.create_engine(f"postgresql+psycopg2://{db_user}@localhost/{db_name}")

query = f"""
SELECT column_name
FROM information_schema.columns
WHERE table_schema = '{db_schema}'
  AND table_name = 'title';
"""

with engine.connect() as conn:
    df = pd.read_sql(query, conn)
    print("📑 Columns in employees.title:")
    print(df)
