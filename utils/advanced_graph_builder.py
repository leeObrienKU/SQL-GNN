import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, HeteroData
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from utils.graph_builder import _standardize, _to_datetime, _latest_by

class AdvancedGraphBuilder:
    def __init__(self):
        self.emp_features = None
        self.dept_features = None
        self.title_features = None
        self.ref_date = None
        
    def _prepare_features(self, employees, departments, titles, salaries, ref_date):
        """Prepare standardized features for all node types"""
        # [Code from original graph_builder, refactored for reuse]
        pass
        
    def create_employee_graph(
        self,
        employees,
        departments,
        dept_emp,
        titles,
        salaries,
        cutoff_date: str = "2002-12-31",
        similarity_threshold: float = 0.7
    ) -> Data:
        """Create homogeneous employee-employee graph
        
        Edges based on:
        1. Department co-working
        2. Similar tenure/age/salary
        3. Similar career paths
        """
        # Build features
        self._prepare_features(employees, departments, titles, salaries, cutoff_date)
        
        # Create employee similarity edges
        edge_index = []
        edge_attr = []
        
        # Department-based connections
        dept_groups = dept_emp.groupby('dept_no')['emp_no'].apply(list)
        for emp_list in dept_groups:
            for i in range(len(emp_list)):
                for j in range(i+1, len(emp_list)):
                    edge_index.append([emp_list[i], emp_list[j]])
                    edge_index.append([emp_list[j], emp_list[i]])
                    edge_attr.extend([1.0, 1.0])  # Department connection weight
        
        # Feature similarity connections
        for i in range(len(employees)):
            for j in range(i+1, len(employees)):
                sim_score = cosine_similarity(self.emp_features[i], self.emp_features[j])
                if sim_score > similarity_threshold:
                    edge_index.append([i, j])
                    edge_index.append([j, i])
                    edge_attr.extend([sim_score, sim_score])
        
        # Create PyG Data object
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        x = torch.from_numpy(self.emp_features).float()
        
        # Labels (1 = leaver)
        y = self._create_attrition_labels(employees, dept_emp, cutoff_date)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data.num_classes = 2
        return data
    
    def create_heterogeneous_graph(
        self,
        employees,
        departments,
        dept_emp,
        titles,
        salaries,
        cutoff_date: str = "2002-12-31"
    ) -> HeteroData:
        """Create heterogeneous graph with multiple node and edge types"""
        # Build features
        self._prepare_features(employees, departments, titles, salaries, cutoff_date)
        
        # Create HeteroData object
        data = HeteroData()
        
        # Add node features
        data['employee'].x = torch.from_numpy(self.emp_features).float()
        data['department'].x = torch.from_numpy(self.dept_features).float()
        data['title'].x = torch.from_numpy(self.title_features).float()
        
        # Add edges
        # Employee -> Department
        emp_dept_edge_index = self._create_emp_dept_edges(employees, dept_emp)
        data['employee', 'works_in', 'department'].edge_index = emp_dept_edge_index
        
        # Employee -> Title
        emp_title_edge_index = self._create_emp_title_edges(employees, titles)
        data['employee', 'has_role', 'title'].edge_index = emp_title_edge_index
        
        # Department -> Title (which titles exist in department)
        dept_title_edge_index = self._create_dept_title_edges(dept_emp, titles)
        data['department', 'contains_role', 'title'].edge_index = dept_title_edge_index
        
        # Add attrition labels
        data['employee'].y = self._create_attrition_labels(employees, dept_emp, cutoff_date)
        
        return data
    
    def create_temporal_graph(
        self,
        employees,
        departments,
        dept_emp,
        titles,
        salaries,
        time_windows: List[str],
        prediction_horizon: str = "6M"
    ) -> List[Data]:
        """Create temporal graph snapshots
        
        Args:
            time_windows: List of dates for snapshots
            prediction_horizon: How far ahead to predict attrition
        """
        graphs = []
        
        for t in time_windows:
            # Create graph snapshot at time t
            snapshot = self._create_graph_snapshot(
                employees, departments, dept_emp, titles, salaries,
                snapshot_date=t,
                label_date=pd.to_datetime(t) + pd.Timedelta(prediction_horizon)
            )
            graphs.append(snapshot)
        
        return graphs
    
    def create_hierarchical_graph(
        self,
        employees,
        departments,
        dept_emp,
        dept_manager,
        titles,
        salaries,
        cutoff_date: str = "2002-12-31"
    ) -> Tuple[Data, Data]:
        """Create hierarchical graph with department and employee levels"""
        # Department-level graph
        dept_graph = self._create_department_graph(departments, dept_emp)
        
        # Employee-level graphs (one per department)
        emp_graphs = {}
        for dept_id in departments['dept_no']:
            dept_employees = dept_emp[dept_emp['dept_no'] == dept_id]['emp_no'].unique()
            emp_graphs[dept_id] = self._create_employee_subgraph(
                employees[employees['emp_no'].isin(dept_employees)],
                titles, salaries, cutoff_date
            )
        
        return dept_graph, emp_graphs
    
    def create_knowledge_graph(
        self,
        employees,
        departments,
        dept_emp,
        titles,
        salaries,
        skills,  # Additional skill data if available
        cutoff_date: str = "2002-12-31"
    ) -> HeteroData:
        """Create knowledge graph with rich semantic relationships"""
        data = HeteroData()
        
        # Node features
        data['employee'].x = torch.from_numpy(self.emp_features).float()
        data['department'].x = torch.from_numpy(self.dept_features).float()
        data['title'].x = torch.from_numpy(self.title_features).float()
        data['skill'].x = torch.from_numpy(self._prepare_skill_features(skills)).float()
        
        # Edge types:
        # - employee -> department (works_in)
        # - employee -> title (has_role)
        # - employee -> skill (has_skill)
        # - title -> skill (requires_skill)
        # - department -> title (offers_role)
        
        # Add edges
        data['employee', 'works_in', 'department'].edge_index = self._create_emp_dept_edges(employees, dept_emp)
        data['employee', 'has_role', 'title'].edge_index = self._create_emp_title_edges(employees, titles)
        if skills is not None:
            data['employee', 'has_skill', 'skill'].edge_index = self._create_emp_skill_edges(employees, skills)
            data['title', 'requires_skill', 'skill'].edge_index = self._create_title_skill_edges(titles, skills)
        
        # Labels
        data['employee'].y = self._create_attrition_labels(employees, dept_emp, cutoff_date)
        
        return data
    
    def _create_attrition_labels(self, employees, dept_emp, cutoff_date):
        """Create binary attrition labels"""
        cutoff = pd.to_datetime(cutoff_date)
        latest_dept = _latest_by(
            dept_emp,
            by_cols=['emp_no'],
            sort_cols=['to_date'],
            keep_cols=['emp_no', 'to_date']
        )
        
        # 1 = leaver (left before cutoff)
        labels = torch.zeros(len(employees), dtype=torch.long)
        for idx, row in latest_dept.iterrows():
            if pd.notna(row['to_date']) and row['to_date'] < cutoff:
                emp_idx = employees[employees['emp_no'] == row['emp_no']].index[0]
                labels[emp_idx] = 1
                
        return labels
    
    def _create_graph_snapshot(self, employees, departments, dept_emp, titles, salaries, snapshot_date, label_date):
        """Create graph snapshot at specific time point"""
        # Filter data to snapshot date
        snapshot_dept_emp = dept_emp[dept_emp['to_date'] >= snapshot_date]
        snapshot_titles = titles[titles['to_date'] >= snapshot_date]
        snapshot_salaries = salaries[salaries['to_date'] >= snapshot_date]
        
        # Build graph
        data = self.create_heterogeneous_graph(
            employees, departments, snapshot_dept_emp,
            snapshot_titles, snapshot_salaries,
            cutoff_date=label_date
        )
        
        return data
