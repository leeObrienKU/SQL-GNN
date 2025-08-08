import psycopg2
import pandas as pd
from tabulate import tabulate

def inspect_database():
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            dbname="empdb",
            user="postgres",
            host="/var/run/postgresql"
        )
        cur = conn.cursor()
        
        print("✅ Connected to PostgreSQL\n")
        
        # 1. List all tables in employees schema
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'employees'
        """)
        tables = [x[0] for x in cur.fetchall()]
        print("📊 Tables in 'employees' schema:")
        print(tabulate([[t] for t in tables], headers=["Table Name"], tablefmt="grid"))
        
        # 2. Check each table's structure
        for table in tables:
            print(f"\n🔍 Structure of employees.{table}:")
            cur.execute(f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_schema = 'employees' AND table_name = '{table}'
            """)
            print(tabulate(cur.fetchall(), headers=["Column", "Data Type"], tablefmt="grid"))
            
            # Show primary/foreign keys
            cur.execute(f"""
                SELECT
                    tc.constraint_name, tc.constraint_type,
                    kcu.column_name, 
                    ccu.table_name AS foreign_table,
                    ccu.column_name AS foreign_column
                FROM 
                    information_schema.table_constraints tc
                    LEFT JOIN information_schema.key_column_usage kcu
                        ON tc.constraint_name = kcu.constraint_name
                    LEFT JOIN information_schema.constraint_column_usage ccu
                        ON ccu.constraint_name = tc.constraint_name
                WHERE 
                    tc.table_schema = 'employees' 
                    AND tc.table_name = '{table}'
            """)
            constraints = cur.fetchall()
            if constraints:
                print("\nConstraints:")
                print(tabulate(constraints, 
                             headers=["Constraint", "Type", "Column", "Foreign Table", "Foreign Column"],
                             tablefmt="grid"))
        
        # 3. Sample data from key tables
        sample_tables = ['employee', 'department', 'salary']
        for table in sample_tables:
            if table in tables:
                print(f"\n📋 Sample data from employees.{table}:")
                cur.execute(f"SELECT * FROM employees.{table} LIMIT 3")
                cols = [desc[0] for desc in cur.description]
                print(tabulate(cur.fetchall(), headers=cols, tablefmt="grid"))
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    inspect_database()