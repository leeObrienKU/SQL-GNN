import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from utils.data_loader import load_employees_db
from utils.graph_builder import create_graph
from torch_geometric.utils import to_networkx

def visualize_graph_simple(cutoff_date="2002-12-31", max_employees=500):
    """Simple graph visualization for quick exploration"""
    print("🔄 Loading data and building graph...")
    
    # Load data
    employees, departments, dept_emp, dept_manager, titles, salaries = load_employees_db()
    
    # Sample for visualization
    if len(employees) > max_employees:
        sample_employees = employees.sample(n=max_employees, random_state=42)
        emp_ids = sample_employees['id'].tolist()
        dept_emp = dept_emp[dept_emp['emp_no'].isin(emp_ids)]
        titles = titles[titles['emp_no'].isin(emp_ids)]
        salaries = salaries[salaries['emp_no'].isin(emp_ids)]
        employees = sample_employees
        print(f"📊 Sampled {max_employees} employees for visualization")
    
    # Build graph
    graph = create_graph(
        employees, departments, dept_emp, dept_manager, 
        titles, salaries, task="attrition", cutoff_date=cutoff_date
    )
    
    # Convert to NetworkX
    nx_graph = to_networkx(graph, to_undirected=True)
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Separate node types
    employee_nodes = [n for n in nx_graph.nodes() if n < len(employees)]
    dept_nodes = [n for n in nx_graph.nodes() if n >= len(employees)]
    
    # Position nodes
    pos = nx.spring_layout(nx_graph, k=1, iterations=50, seed=42)
    
    # Draw edges
    nx.draw_networkx_edges(nx_graph, pos, alpha=0.2, edge_color='gray', width=0.5)
    
    # Draw employee nodes (colored by attrition)
    if hasattr(graph, 'y'):
        attrition_labels = graph.y[:len(employees)]
        colors = ['red' if label == 1 else 'blue' for label in attrition_labels]
        nx.draw_networkx_nodes(nx_graph, pos, nodelist=employee_nodes, 
                             node_color=colors, node_size=50, alpha=0.8)
    else:
        nx.draw_networkx_nodes(nx_graph, pos, nodelist=employee_nodes, 
                             node_color='blue', node_size=50, alpha=0.8)
    
    # Draw department nodes
    nx.draw_networkx_nodes(nx_graph, pos, nodelist=dept_nodes, 
                         node_color='green', node_size=100, alpha=0.8)
    
    # Add department labels
    dept_labels = {n: f"D{n-len(employees)}" for n in dept_nodes}
    nx.draw_networkx_labels(nx_graph, pos, labels=dept_labels, font_size=8)
    
    plt.title(f"Employee-Department Graph (Cutoff: {cutoff_date})\nRed=Leavers, Blue=Stayers, Green=Departments", fontsize=14)
    plt.axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.8, label='Leavers'),
        Patch(facecolor='blue', alpha=0.8, label='Stayers'),
        Patch(facecolor='green', alpha=0.8, label='Departments')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    return plt

if __name__ == "__main__":
    # Create visualization
    plt = visualize_graph_simple()
    plt.show()
    
    # Save high-quality version
    plt.savefig("employee_department_graph.png", dpi=300, bbox_inches='tight')
    print("✅ Graph saved as employee_department_graph.png")
