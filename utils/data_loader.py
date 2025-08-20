from sqlalchemy import create_engine
import pandas as pd
from sqlalchemy.exc import SQLAlchemyError

def load_employees_db():
    """Load data from PostgreSQL with schema-aligned column names"""
    try:
        # Create SQLAlchemy engine
        engine = create_engine('postgresql+psycopg2://postgres@/empdb?host=/var/run/postgresql')

        # Define queries matching your actual schema
        queries = {
            'employees': """
                SELECT 
                    id AS emp_no,
                    birth_date,
                    first_name,
                    last_name,
                    gender,
                    hire_date
                FROM employees.employee
            """,
            'departments': """
                SELECT 
                    id AS dept_no,
                    dept_name
                FROM employees.department
            """,
            'dept_emp': """
                SELECT
                    employee_id AS emp_no,
                    department_id AS dept_no,
                    from_date,
                    to_date
                FROM employees.department_employee
            """,
            'dept_manager': """
                SELECT
                    employee_id AS emp_no,
                    department_id AS dept_no,
                    from_date,
                    to_date
                FROM employees.department_manager
            """,
            'salaries': """
                SELECT
                    employee_id AS emp_no,
                    amount AS salary,
                    from_date,
                    to_date
                FROM employees.salary
            """,
            'titles': """
                SELECT
                    employee_id AS emp_no,
                    title,
                    from_date,
                    to_date
                FROM employees.title
            """
        }

        # Execute queries
        data = {}
        for name, query in queries.items():
            try:
                data[name] = pd.read_sql(query, engine)
                if data[name].empty:
                    print(f"⚠️  Empty result for {name}")
            except SQLAlchemyError as e:
                print(f"⚠️  Couldn't load {name}: {str(e)}")
                data[name] = pd.DataFrame(columns=query.split('SELECT ')[1].split('FROM')[0].replace('\n', '').replace(' ', '').split(','))

        # Debug output
        print("\n📦 Loaded Data Summary:")
        for name, df in data.items():
            print(f"{name:12s}: {df.shape[0]:6d} rows | Columns: {list(df.columns)}")

        return (
            data['employees'],
            data['departments'],
            data['dept_emp'],
            data['dept_manager'],
            data['titles'],
            data['salaries']
        )

    except Exception as e:
        print(f"\n❌ Critical Error: {str(e)}")
        raise