import networkx as nx
import pandas as pd

def create_graph(employees, departments, dept_emp, dept_manager, titles, salaries):
    G = nx.Graph()
    
    # Add employee nodes
    for _, row in employees.iterrows():
        G.add_node(f"emp_{row['emp_no']}", type='employee', **row.to_dict())
    
    # Add department nodes
    for _, row in departments.iterrows():
        G.add_node(f"dept_{row['dept_no']}", type='department', **row.to_dict())
    
    # Add relationships
    for _, row in dept_emp.iterrows():
        G.add_edge(f"emp_{row['emp_no']}", f"dept_{row['dept_no']}",
                  type='works_in', from_date=row['from_date'], to_date=row['to_date'])
    
    return G
