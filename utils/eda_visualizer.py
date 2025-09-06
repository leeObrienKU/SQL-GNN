import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

class EDAVisualizer:
    def __init__(self, output_dir="/Users/lee/Edge/projects/gnn-sql/gnn_sql_project/experiment_logs/eda"):
        """Initialize visualizer with output directory"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Set style
        plt.style.use('default')
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['axes.facecolor'] = 'white'
        
        # Set color palette
        self.colors = ['#FF7F50', '#87CEEB', '#98FB98', '#DDA0DD', '#F0E68C']
    
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
        ax1.hist(salary_data['amount'], bins=50, color=self.colors[0], alpha=0.7)
        ax1.set_title('Salary Distribution')
        ax1.set_xlabel('Salary Amount')
        ax1.grid(True)
        
        # 2. Box plot by department
        dept_data = [group['amount'].values for name, group in salary_data.groupby('dept_name')]
        ax2.boxplot(dept_data, labels=salary_data['dept_name'].unique())
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        ax2.set_title('Salary Distribution by Department')
        ax2.grid(True)
        
        # 3. Salary trends over time
        yearly_avg = salary_data.groupby('year')['amount'].mean()
        ax3.plot(yearly_avg.index, yearly_avg.values, color=self.colors[2], marker='o')
        ax3.set_title('Average Salary Trend')
        ax3.grid(True)
        
        # 4. Salary growth distribution
        growth = salary_data.groupby('employee_id')['amount'].pct_change()
        ax4.hist(growth.dropna(), bins=50, color=self.colors[3], alpha=0.7)
        ax4.set_title('Salary Growth Distribution')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"salary_analysis_{self.timestamp}.png")
        plt.close()
    
    def plot_tenure_analysis(self, tenure_data):
        """Plot tenure-related visualizations"""
        plt.figure(figsize=(15, 10))
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Overall tenure distribution
        ax1.hist(tenure_data['tenure_years'], bins=30, color=self.colors[0], alpha=0.7)
        ax1.set_title('Employee Tenure Distribution')
        ax1.grid(True)
        
        # 2. Tenure by department
        dept_tenure = [group['tenure_years'].values for name, group in tenure_data.groupby('dept_name')]
        ax2.boxplot(dept_tenure, labels=tenure_data['dept_name'].unique())
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        ax2.set_title('Tenure by Department')
        ax2.grid(True)
        
        # 3. Tenure vs. Salary
        ax3.scatter(tenure_data['tenure_years'], tenure_data['salary'], 
                   alpha=0.5, color=self.colors[2])
        ax3.set_title('Tenure vs. Salary')
        ax3.grid(True)
        
        # 4. Attrition by tenure
        tenure_groups = tenure_data.groupby('tenure_bucket')
        attrition_rates = tenure_groups['attrition_rate'].mean()
        ax4.bar(range(len(attrition_rates)), attrition_rates.values, 
                color=self.colors[3], alpha=0.7)
        ax4.set_xticks(range(len(attrition_rates)))
        ax4.set_xticklabels(attrition_rates.index, rotation=45)
        ax4.set_title('Attrition Rate by Tenure')
        ax4.grid(True)
        
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
        y_pos = np.arange(len(title_counts))
        ax1.barh(y_pos, title_counts.values, color=self.colors[0], alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(title_counts.index)
        ax1.set_title('Title Distribution')
        ax1.grid(True)
        
        # 2. Title changes over time
        title_changes = title_data.groupby('year').size()
        ax2.plot(title_changes.index, title_changes.values, 
                 color=self.colors[1], marker='o')
        ax2.set_title('Title Changes Over Time')
        ax2.grid(True)
        
        # 3. Title duration distribution
        ax3.hist(title_data['duration_years'], bins=30, 
                 color=self.colors[2], alpha=0.7)
        ax3.set_title('Title Duration Distribution')
        ax3.grid(True)
        
        # 4. Title changes by department
        dept_changes = [group['title_changes'].values for name, group in title_data.groupby('dept_name')]
        ax4.boxplot(dept_changes, labels=title_data['dept_name'].unique())
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
        ax4.set_title('Title Changes by Department')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"title_analysis_{self.timestamp}.png")
        plt.close()
    
    def plot_correlation_matrix(self, feature_data):
        """Plot feature correlation matrix"""
        plt.figure(figsize=(12, 10))
        
        # Calculate correlations
        corr = feature_data.corr()
        
        # Create heatmap
        im = plt.imshow(corr, cmap='coolwarm', aspect='auto')
        plt.colorbar(im)
        
        # Add correlation values
        for i in range(len(corr)):
            for j in range(len(corr)):
                plt.text(j, i, f'{corr.iloc[i, j]:.2f}',
                        ha='center', va='center')
        
        # Add labels
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
        plt.yticks(range(len(corr.columns)), corr.columns)
        
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(self.output_dir / f"correlation_matrix_{self.timestamp}.png")
        plt.close()
    
    def plot_class_imbalance(self, stayers, leavers, class_stats):
        """Visualize class imbalance"""
        plt.figure(figsize=(15, 10))
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Class distribution pie chart
        ax1.pie([stayers, leavers], labels=['Stayers', 'Leavers'], autopct='%1.1f%%',
                colors=[self.colors[0], self.colors[1]])
        ax1.set_title('Class Distribution')
        
        # 2. Age distribution by class
        stayer_stats = next(stat for stat in class_stats if stat[0] == 'Stayer')
        leaver_stats = next(stat for stat in class_stats if stat[0] == 'Leaver')
        
        ax2.bar(['Stayers', 'Leavers'], [stayer_stats[3], leaver_stats[3]], 
                color=[self.colors[0], self.colors[1]])
        ax2.set_title('Average Age by Class')
        ax2.set_ylabel('Age (years)')
        
        # 3. Tenure distribution by class
        ax3.bar(['Stayers', 'Leavers'], [stayer_stats[4], leaver_stats[4]], 
                color=[self.colors[0], self.colors[1]])
        ax3.set_title('Average Tenure by Class')
        ax3.set_ylabel('Tenure (years)')
        
        # 4. Gender distribution by class
        x = np.arange(2)
        width = 0.35
        ax4.bar(x - width/2, [stayer_stats[5], leaver_stats[5]], width, label='Male',
                color=self.colors[2])
        ax4.bar(x + width/2, [stayer_stats[6], leaver_stats[6]], width, label='Female',
                color=self.colors[3])
        ax4.set_xticks(x)
        ax4.set_xticklabels(['Stayers', 'Leavers'])
        ax4.set_title('Gender Distribution by Class')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"class_imbalance_{self.timestamp}.png")
        plt.close()

    def plot_missing_data(self, data):
        """Visualize missing data patterns"""
        plt.figure(figsize=(10, 6))
        
        # Calculate missing percentages
        missing = (data.isnull().sum() / len(data)) * 100
        missing = missing[missing > 0].sort_values(ascending=True)
        
        # Create bar plot
        y_pos = np.arange(len(missing))
        plt.barh(y_pos, missing.values, color=self.colors[0], alpha=0.7)
        plt.yticks(y_pos, missing.index)
        plt.title('Missing Data Analysis')
        plt.xlabel('Percentage Missing')
        plt.grid(True)
        
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
                <h2>Class Imbalance Analysis</h2>
                <img src="class_imbalance_{self.timestamp}.png" alt="Class Imbalance">
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