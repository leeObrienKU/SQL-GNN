import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from utils.data_loader import load_employees_db
from utils.graph_builder import create_graph
import torch
from torch_geometric.utils import to_networkx
import streamlit as st
import streamlit.components.v1 as components

class InteractiveGraphVisualizer:
    def __init__(self):
        self.graph = None
        self.nx_graph = None
        self.employee_data = None
        self.department_data = None
        
    def load_and_build_graph(self, cutoff_date="2002-12-31", max_employees=1000):
        """Load data and build graph with sample size limit for visualization"""
        print("🔄 Loading database and building graph...")
        
        # Load data
        employees, departments, dept_emp, dept_manager, titles, salaries = load_employees_db()
        
        # Sample employees for visualization (to avoid overwhelming the plot)
        if len(employees) > max_employees:
            sample_employees = employees.sample(n=max_employees, random_state=42)
            emp_ids = sample_employees['id'].tolist()
            
            # Filter related data
            dept_emp = dept_emp[dept_emp['emp_no'].isin(emp_ids)]
            titles = titles[titles['emp_no'].isin(emp_ids)]
            salaries = salaries[salaries['emp_no'].isin(emp_ids)]
            employees = sample_employees
            
            print(f"📊 Sampled {max_employees} employees for visualization")
        
        # Build graph
        self.graph = create_graph(
            employees, departments, dept_emp, dept_manager, 
            titles, salaries, task="attrition", cutoff_date=cutoff_date
        )
        
        # Convert to NetworkX for visualization
        self.nx_graph = to_networkx(self.graph, to_undirected=True)
        
        # Store data for node attributes
        self.employee_data = employees
        self.department_data = departments
        
        print(f"✅ Graph built with {self.nx_graph.number_of_nodes()} nodes and {self.nx_graph.number_of_edges()} edges")
        return self.nx_graph
    
    def create_static_visualization(self, figsize=(15, 10), node_size=100, alpha=0.7):
        """Create static matplotlib visualization"""
        plt.figure(figsize=figsize)
        
        # Separate employee and department nodes
        employee_nodes = [n for n in self.nx_graph.nodes() if n < len(self.employee_data)]
        dept_nodes = [n for n in self.nx_graph.nodes() if n >= len(self.employee_data)]
        
        # Position nodes using spring layout
        pos = nx.spring_layout(self.nx_graph, k=1, iterations=50, seed=42)
        
        # Draw edges
        nx.draw_networkx_edges(self.nx_graph, pos, alpha=0.3, edge_color='gray')
        
        # Draw employee nodes (colored by attrition status)
        if hasattr(self.graph, 'y'):
            attrition_labels = self.graph.y[:len(self.employee_data)]
            colors = ['red' if label == 1 else 'blue' for label in attrition_labels]
            nx.draw_networkx_nodes(self.nx_graph, pos, nodelist=employee_nodes, 
                                 node_color=colors, node_size=node_size, alpha=alpha)
        else:
            nx.draw_networkx_nodes(self.nx_graph, pos, nodelist=employee_nodes, 
                                 node_color='blue', node_size=node_size, alpha=alpha)
        
        # Draw department nodes
        nx.draw_networkx_nodes(self.nx_graph, pos, nodelist=dept_nodes, 
                             node_color='green', node_size=node_size*2, alpha=alpha)
        
        # Add labels for departments
        dept_labels = {n: f"Dept {n-len(self.employee_data)}" for n in dept_nodes}
        nx.draw_networkx_labels(self.nx_graph, pos, labels=dept_labels, font_size=8)
        
        plt.title("Employee-Department Bipartite Graph\n(Red=Leavers, Blue=Stayers, Green=Departments)", fontsize=14)
        plt.axis('off')
        return plt
    
    def create_interactive_plotly(self):
        """Create interactive Plotly visualization"""
        # Separate employee and department nodes
        employee_nodes = [n for n in self.nx_graph.nodes() if n < len(self.employee_data)]
        dept_nodes = [n for n in self.nx_graph.nodes() if n >= len(self.employee_data)]
        
        # Position nodes using spring layout
        pos = nx.spring_layout(self.nx_graph, k=1, iterations=50, seed=42)
        
        # Prepare node data
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        
        # Employee nodes
        for node in employee_nodes:
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Get employee info
            emp_idx = node
            if emp_idx < len(self.employee_data):
                emp = self.employee_data.iloc[emp_idx]
                gender = emp.get('gender', 'Unknown')
                hire_date = emp.get('hire_date', 'Unknown')
                attrition = "Leaver" if (hasattr(self.graph, 'y') and self.graph.y[emp_idx] == 1) else "Stayer"
                
                node_text.append(f"Employee {emp_idx}<br>Gender: {gender}<br>Hire: {hire_date}<br>Status: {attrition}")
                node_color.append('red' if attrition == "Leaver" else 'blue')
                node_size.append(15)
            else:
                node_text.append(f"Employee {emp_idx}")
                node_color.append('gray')
                node_size.append(15)
        
        # Department nodes
        for node in dept_nodes:
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            dept_idx = node - len(self.employee_data)
            if dept_idx < len(self.department_data):
                dept = self.department_data.iloc[dept_idx]
                dept_name = dept.get('dept_name', f'Dept {dept_idx}')
                node_text.append(f"Department: {dept_name}")
            else:
                node_text.append(f"Department {dept_idx}")
            
            node_color.append('green')
            node_size.append(25)
        
        # Prepare edge data
        edge_x = []
        edge_y = []
        
        for edge in self.nx_graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Create the interactive plot
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=0.5, color='gray'),
            hoverinfo='none',
            showlegend=False
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(
                size=node_size,
                color=node_color,
                opacity=0.8,
                line=dict(width=2, color='white')
            ),
            text=node_text,
            hoverinfo='text',
            showlegend=False
        ))
        
        # Update layout
        fig.update_layout(
            title="Interactive Employee-Department Graph<br><sub>Hover over nodes for details | Red=Leavers, Blue=Stayers, Green=Departments</sub>",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        return fig
    
    def create_network_analysis_dashboard(self):
        """Create comprehensive network analysis dashboard"""
        # Calculate network metrics
        metrics = {
            "Total Nodes": self.nx_graph.number_of_nodes(),
            "Total Edges": self.nx_graph.number_of_edges(),
            "Employee Nodes": len([n for n in self.nx_graph.nodes() if n < len(self.employee_data)]),
            "Department Nodes": len([n for n in self.nx_graph.nodes() if n >= len(self.employee_data)]),
            "Average Degree": round(self.nx_graph.number_of_edges() * 2 / self.nx_graph.number_of_nodes(), 2),
            "Density": round(nx.density(self.nx_graph), 4)
        }
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Network Overview", "Degree Distribution", "Node Types", "Edge Distribution"),
            specs=[[{"type": "table"}, {"type": "bar"}],
                   [{"type": "pie"}, {"type": "histogram"}]]
        )
        
        # Network overview table
        fig.add_trace(
            go.Table(
                header=dict(values=["Metric", "Value"]),
                cells=dict(values=[list(metrics.keys()), list(metrics.values())])
            ),
            row=1, col=1
        )
        
        # Degree distribution
        degrees = [d for n, d in self.nx_graph.degree()]
        fig.add_trace(
            go.Histogram(x=degrees, nbinsx=20, name="Degree Distribution"),
            row=1, col=2
        )
        
        # Node types pie chart
        emp_count = len([n for n in self.nx_graph.nodes() if n < len(self.employee_data)])
        dept_count = len([n for n in self.nx_graph.nodes() if n >= len(self.employee_data)])
        fig.add_trace(
            go.Pie(labels=["Employees", "Departments"], values=[emp_count, dept_count]),
            row=2, col=1
        )
        
        # Edge distribution (if we have edge weights or types)
        fig.add_trace(
            go.Histogram(x=[1]*self.nx_graph.number_of_edges(), nbinsx=10, name="Edge Distribution"),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Network Analysis Dashboard")
        return fig

def create_streamlit_app():
    """Create Streamlit web application for interactive graph exploration"""
    st.set_page_config(page_title="GNN Graph Visualizer", layout="wide")
    
    st.title("🎯 Interactive GNN Graph Visualizer")
    st.markdown("Explore the employee-department bipartite graph structure used in your GNN model")
    
    # Sidebar controls
    st.sidebar.header("Graph Configuration")
    cutoff_date = st.sidebar.text_input("Cutoff Date", value="2002-12-31", help="Date to separate stayers from leavers")
    max_employees = st.sidebar.slider("Max Employees to Visualize", min_value=100, max_value=2000, value=500, step=100)
    
    if st.sidebar.button("🔄 Build Graph"):
        with st.spinner("Building graph..."):
            visualizer = InteractiveGraphVisualizer()
            graph = visualizer.load_and_build_graph(cutoff_date, max_employees)
            
            st.session_state.visualizer = visualizer
            st.success(f"Graph built successfully! {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    if 'visualizer' in st.session_state:
        visualizer = st.session_state.visualizer
        
        # Tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["📊 Interactive Graph", "📈 Network Analysis", "🖼️ Static View"])
        
        with tab1:
            st.header("Interactive Graph Visualization")
            st.markdown("Hover over nodes to see details. Red nodes = Leavers, Blue nodes = Stayers, Green nodes = Departments")
            
            fig = visualizer.create_interactive_plotly()
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.header("Network Analysis Dashboard")
            st.markdown("Comprehensive metrics and analysis of the graph structure")
            
            dashboard = visualizer.create_network_analysis_dashboard()
            st.plotly_chart(dashboard, use_container_width=True)
        
        with tab3:
            st.header("Static Graph View")
            st.markdown("Traditional matplotlib visualization for publication")
            
            static_plot = visualizer.create_static_visualization()
            st.pyplot(static_plot)
            
            # Download button
            if st.button("📥 Download Static Plot"):
                static_plot.savefig("graph_visualization.png", dpi=300, bbox_inches='tight')
                st.success("Plot saved as graph_visualization.png")

if __name__ == "__main__":
    # For command line usage
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "streamlit":
        # Run Streamlit app
        import subprocess
        subprocess.run(["streamlit", "run", __file__])
    else:
        # Command line usage
        visualizer = InteractiveGraphVisualizer()
        graph = visualizer.load_and_build_graph()
        
        print("🎯 Graph visualization options:")
        print("1. Static matplotlib plot")
        print("2. Interactive Plotly plot")
        print("3. Network analysis dashboard")
        print("4. Streamlit web app")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == "1":
            plt = visualizer.create_static_visualization()
            plt.show()
        elif choice == "2":
            fig = visualizer.create_interactive_plotly()
            fig.show()
        elif choice == "3":
            dashboard = visualizer.create_network_analysis_dashboard()
            dashboard.show()
        elif choice == "4":
            print("Run: python utils/graph_visualizer.py streamlit")
        else:
            print("Invalid choice. Running static visualization...")
            plt = visualizer.create_static_visualization()
            plt.show()
