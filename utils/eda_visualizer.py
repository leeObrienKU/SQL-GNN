import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

class EDAVisualizer:
    def __init__(self, output_dir="/Users/lee/Edge/projects/gnn-sql/gnn_sql_project/experiment_logs/eda"):
        """Initialize visualizer with output directory"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Set style
        plt.style.use('seaborn')
        sns.set_palette("husl")
    
    def plot_entity_relationships(self, tables_info):
        """Create entity relationship diagram"""
        plt.figure(figsize=(12, 8))
        
        # Create graph using networkx
        import networkx as nx
        G = nx.Graph()
        
        # Add nodes (tables)
        for table, info in tables_info.items():
            G.add_node(table)
        
        # Add edges (relationships)
        relationships = [
            ('employee', 'department_employee'),
            ('department', 'department_employee'),
            ('employee', 'salary'),
            ('employee', 'title'),
            ('department', 'department_manager'),
            ('employee', 'department_manager')
        ]
        G.add_edges_from(relationships)
        
        # Draw graph
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=2000, font_size=10, font_weight='bold')
        
        plt.title("Entity Relationship Diagram")
        plt.savefig(self.output_dir / f"entity_relationships_{self.timestamp}.png")
        plt.close()
    
    def plot_salary_distribution(self, salary_data):
        """Plot salary distribution and trends"""
        plt.figure(figsize=(15, 10))
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Overall distribution
        sns.histplot(salary_data['amount'], bins=50, ax=ax1)
        ax1.set_title('Salary Distribution')
        ax1.set_xlabel('Salary Amount')
        
        # 2. Box plot by department
        sns.boxplot(x='dept_name', y='amount', data=salary_data, ax=ax2)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        ax2.set_title('Salary Distribution by Department')
        
        # 3. Salary trends over time
        sns.lineplot(x='year', y='avg_salary', data=salary_data.groupby('year').agg({'amount': 'mean'}).reset_index(), ax=ax3)
        ax3.set_title('Average Salary Trend')
        
        # 4. Salary growth distribution
        sns.histplot(salary_data.groupby('employee_id')['amount'].pct_change(), bins=50, ax=ax4)
        ax4.set_title('Salary Growth Distribution')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"salary_analysis_{self.timestamp}.png")
        plt.close()
    
    def plot_tenure_analysis(self, tenure_data):
        """Plot tenure-related visualizations"""
        plt.figure(figsize=(15, 10))
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Overall tenure distribution
        sns.histplot(tenure_data['tenure_years'], bins=30, ax=ax1)
        ax1.set_title('Employee Tenure Distribution')
        
        # 2. Tenure by department
        sns.boxplot(x='dept_name', y='tenure_years', data=tenure_data, ax=ax2)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        ax2.set_title('Tenure by Department')
        
        # 3. Tenure vs. Salary
        sns.scatterplot(x='tenure_years', y='salary', data=tenure_data, ax=ax3)
        ax3.set_title('Tenure vs. Salary')
        
        # 4. Attrition by tenure
        sns.barplot(x='tenure_bucket', y='attrition_rate', data=tenure_data, ax=ax4)
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
        ax4.set_title('Attrition Rate by Tenure')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"tenure_analysis_{self.timestamp}.png")
        plt.close()
    
    def plot_title_history(self, title_data):
        """Plot title change patterns"""
        plt.figure(figsize=(15, 10))
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Title distribution
        title_counts = title_data['title'].value_counts()
        sns.barplot(x=title_counts.values, y=title_counts.index, ax=ax1)
        ax1.set_title('Title Distribution')
        
        # 2. Title changes over time
        title_changes = title_data.groupby('year').size()
        sns.lineplot(x=title_changes.index, y=title_changes.values, ax=ax2)
        ax2.set_title('Title Changes Over Time')
        
        # 3. Title duration distribution
        sns.histplot(title_data['duration_years'], bins=30, ax=ax3)
        ax3.set_title('Title Duration Distribution')
        
        # 4. Title changes by department
        sns.boxplot(x='dept_name', y='title_changes', data=title_data, ax=ax4)
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
        ax4.set_title('Title Changes by Department')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"title_analysis_{self.timestamp}.png")
        plt.close()
    
    def plot_correlation_matrix(self, feature_data):
        """Plot feature correlation matrix"""
        plt.figure(figsize=(12, 10))
        
        # Calculate correlations
        corr = feature_data.corr()
        
        # Create heatmap
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0,
                   fmt='.2f', square=True, linewidths=0.5)
        
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(self.output_dir / f"correlation_matrix_{self.timestamp}.png")
        plt.close()
    
    def plot_missing_data(self, data):
        """Visualize missing data patterns"""
        plt.figure(figsize=(10, 6))
        
        # Calculate missing percentages
        missing = (data.isnull().sum() / len(data)) * 100
        missing = missing[missing > 0].sort_values(ascending=True)
        
        # Create bar plot
        sns.barplot(x=missing.values, y=missing.index)
        plt.title('Missing Data Analysis')
        plt.xlabel('Percentage Missing')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"missing_data_{self.timestamp}.png")
        plt.close()
    
    def create_html_report(self, title="EDA Report"):
        """Create an HTML report with all visualizations"""
        html_content = f"""
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .section {{ margin-bottom: 30px; }}
                img {{ max-width: 100%; height: auto; margin: 10px 0; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            <div class="section">
                <h2>Entity Relationships</h2>
                <img src="entity_relationships_{self.timestamp}.png" alt="Entity Relationships">
            </div>
            <div class="section">
                <h2>Salary Analysis</h2>
                <img src="salary_analysis_{self.timestamp}.png" alt="Salary Analysis">
            </div>
            <div class="section">
                <h2>Tenure Analysis</h2>
                <img src="tenure_analysis_{self.timestamp}.png" alt="Tenure Analysis">
            </div>
            <div class="section">
                <h2>Title Analysis</h2>
                <img src="title_analysis_{self.timestamp}.png" alt="Title Analysis">
            </div>
            <div class="section">
                <h2>Feature Correlations</h2>
                <img src="correlation_matrix_{self.timestamp}.png" alt="Correlation Matrix">
            </div>
            <div class="section">
                <h2>Missing Data Analysis</h2>
                <img src="missing_data_{self.timestamp}.png" alt="Missing Data">
            </div>
        </body>
        </html>
        """
        
        with open(self.output_dir / f"eda_report_{self.timestamp}.html", 'w') as f:
            f.write(html_content)
