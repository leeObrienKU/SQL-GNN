
import pandas as pd
import torch
from torch_geometric.data import Data

def create_graph(employees, departments, dept_emp, dept_manager, titles, salaries):
    # Create mappings
    emp_ids = employees['emp_no'].unique()
    dept_ids = departments['dept_no'].unique()

    emp_id_map = {emp_id: idx for idx, emp_id in enumerate(emp_ids)}
    dept_id_map = {dept_id: idx + len(emp_ids) for idx, dept_id in enumerate(dept_ids)}

    # Total nodes = employees + departments
    num_nodes = len(emp_ids) + len(dept_ids)
    node_features = torch.zeros((num_nodes, 4))  # Adjust dimension if needed

    # Example: encode gender and birth_year
    for _, row in employees.iterrows():
        idx = emp_id_map[row['emp_no']]
        birth_year = int(row['birth_date'].split('-')[0])
        gender = 0 if row['gender'] == 'M' else 1
        node_features[idx, 0] = birth_year
        node_features[idx, 1] = gender

    for _, row in departments.iterrows():
        idx = dept_id_map[row['dept_no']]
        node_features[idx, 2] = 1  # department node type marker

    # Create edge_index for works_in relation
    edges = []
    for _, row in dept_emp.iterrows():
        emp_idx = emp_id_map.get(row['emp_no'])
        dept_idx = dept_id_map.get(row['dept_no'])
        if emp_idx is not None and dept_idx is not None:
            edges.append([emp_idx, dept_idx])
            edges.append([dept_idx, emp_idx])  # undirected

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Fake labels and masks for example purposes
    y = torch.zeros(num_nodes, dtype=torch.long)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:int(0.6 * num_nodes)] = True
    val_mask[int(0.6 * num_nodes):int(0.8 * num_nodes)] = True
    test_mask[int(0.8 * num_nodes):] = True

    data = Data(x=node_features, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data
