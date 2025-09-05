import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, HeteroData
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from utils.graph_builder import _standardize, _to_datetime, _latest_by
from tqdm import tqdm
import gc

CURRENT_DATE = '9999-01-01'

class AdvancedGraphBuilder:
    def __init__(self):
        self.emp_features = None
        self.dept_features = None
        self.title_features = None
        self.ref_date = None

    def _safe_to_datetime(self, date_str):
        """Safely convert date string to datetime, handling 9999-01-01 special case"""
        if str(date_str) == CURRENT_DATE:
            # Use a far future date that's within pandas limits
            return pd.Timestamp.max
        return pd.to_datetime(date_str)

    def _prepare_features(self, employees, departments, dept_emp, titles, salaries, ref_date):
        """Prepare standardized features for all node types"""
        print("Preparing features...")
        self.ref_date = self._safe_to_datetime(ref_date)
        
        # Employee features (same as in graph_builder.py)
        emp_id_col = "emp_no" if "emp_no" in employees.columns else "id"
        dept_id_col = "dept_no" if "dept_no" in departments.columns else "id"
        
        # Calculate age and tenure
        employees = employees.copy()
        employees['birth_date'] = pd.to_datetime(employees['birth_date'])
        employees['hire_date'] = pd.to_datetime(employees['hire_date'])
        ref_date = self._safe_to_datetime(ref_date)
        
        employees['age_years'] = (ref_date - employees['birth_date']).dt.days / 365.25
        employees['tenure_years'] = (ref_date - employees['hire_date']).dt.days / 365.25
        
        # Get latest salary
        print("Processing salary data...")
        s_latest = _latest_by(
            salaries,
            by_cols=[emp_id_col],
            sort_cols=['to_date'],
            keep_cols=[emp_id_col, 'salary']
        )
        s_latest = s_latest.rename(columns={'salary': 'curr_salary'})
        
        # Get latest department
        print("Processing department data...")
        d_latest = _latest_by(
            dept_emp,
            by_cols=[emp_id_col],
            sort_cols=['to_date'],
            keep_cols=[emp_id_col, dept_id_col]
        )
        d_latest = d_latest.rename(columns={dept_id_col: 'dept_latest'})
        
        # Get latest title
        print("Processing title data...")
        t_latest = _latest_by(
            titles,
            by_cols=[emp_id_col],
            sort_cols=['to_date'],
            keep_cols=[emp_id_col, 'title']
        )
        t_latest['title_code'] = t_latest['title'].astype("category").cat.codes.astype("int64")
        t_latest = t_latest[[emp_id_col, 'title_code']]
        
        # Calculate salary growth
        print("Calculating salary growth...")
        if len(salaries) > 1:
            salary_growth = salaries.groupby(emp_id_col)['salary'].apply(
                lambda x: (x.iloc[-1] - x.iloc[0]) / max(x.iloc[0], 1e-6) if len(x) > 1 else 0.0
            ).reset_index()
            salary_growth.columns = [emp_id_col, 'salary_growth']
        else:
            salary_growth = pd.DataFrame({emp_id_col: employees[emp_id_col], 'salary_growth': 0.0})
        
        # Assemble employee features
        print("Assembling employee features...")
        emp_feat = employees[[emp_id_col, "age_years", "tenure_years"]].copy()
        emp_feat = emp_feat.merge(s_latest[[emp_id_col, "curr_salary"]], on=emp_id_col, how="left")
        emp_feat = emp_feat.merge(salary_growth, on=emp_id_col, how="left")
        emp_feat = emp_feat.merge(t_latest, on=emp_id_col, how="left")
        emp_feat = emp_feat.merge(d_latest[[emp_id_col, "dept_latest"]], on=emp_id_col, how="left")
        
        # Fill missing values
        for c in ["age_years", "tenure_years", "curr_salary", "salary_growth", "title_code"]:
            if c not in emp_feat.columns:
                emp_feat[c] = 0.0
        emp_feat[["age_years", "tenure_years", "curr_salary", "salary_growth"]] = \
            emp_feat[["age_years", "tenure_years", "curr_salary", "salary_growth"]].fillna(0.0)
        emp_feat["title_code"] = emp_feat["title_code"].fillna(0).astype("int64")
        
        # Department one-hot encoding
        print("Creating department one-hot encoding...")
        dept_list = sorted(departments[dept_id_col].astype(str).unique().tolist())
        dept_to_class = {d: i for i, d in enumerate(dept_list)}
        K = len(dept_list)
        
        emp_feat["dept_latest"] = emp_feat["dept_latest"].astype(str)
        emp_dept_idx = emp_feat["dept_latest"].map(dept_to_class).fillna(-1).astype("int64")
        emp_dept_onehot = np.zeros((emp_feat.shape[0], K), dtype=np.float32)
        valid_mask = emp_dept_idx.values >= 0
        emp_dept_onehot[np.where(valid_mask)[0], emp_dept_idx.values[valid_mask]] = 1.0
        
        # Standardize numeric features
        print("Standardizing numeric features...")
        numeric_cols = ["age_years", "tenure_years", "curr_salary", "salary_growth", "title_code"]
        emp_feat_std = _standardize(emp_feat[numeric_cols].copy(), numeric_cols)
        
        # Final employee features: numeric(5) + dept_onehot(K)
        self.emp_features = np.hstack([emp_feat_std.values.astype(np.float32), emp_dept_onehot])
        
        # Department features: zeros(5) + identity(K)
        dept_identity = np.eye(K, dtype=np.float32)
        dept_numeric_zeros = np.zeros((len(departments), len(numeric_cols)), dtype=np.float32)
        self.dept_features = np.hstack([dept_numeric_zeros, dept_identity])
        
        # Title features: simple encoding
        title_codes = t_latest['title_code'].values
        self.title_features = np.eye(len(np.unique(title_codes)), dtype=np.float32)
        
        print("Feature preparation complete!")
        gc.collect()  # Force garbage collection

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
        """Create homogeneous employee-employee graph"""
        print("\nCreating employee graph...")
        
        # Build features
        self._prepare_features(employees, departments, dept_emp, titles, salaries, cutoff_date)
        
        print("\nCreating department-based connections...")
        edge_index_list = []
        edge_attr_list = []
        
        # Process one department at a time
        dept_groups = dept_emp.groupby('dept_no')
        total_depts = len(dept_groups)
        
        for dept_idx, (dept_id, dept_data) in enumerate(tqdm(dept_groups, total=total_depts)):
            # Get employees in this department
            dept_employees = dept_data['emp_no'].values
            
            if len(dept_employees) > 1:
                # Create indices for this department's employees
                emp_indices = []
                for emp_no in dept_employees:
                    try:
                        idx = employees[employees['emp_no'] == emp_no].index[0]
                        emp_indices.append(idx)
                    except IndexError:
                        continue
                
                if len(emp_indices) > 1:
                    emp_indices = np.array(emp_indices)
                    
                    # Create edges between all pairs in this department (vectorized)
                    idx1, idx2 = np.meshgrid(emp_indices, emp_indices)
                    mask = idx1 < idx2  # Upper triangular only to avoid duplicates
                    
                    # Add edges in both directions
                    edges = np.vstack([
                        np.concatenate([idx1[mask], idx2[mask]]),  # forward edges
                        np.concatenate([idx2[mask], idx1[mask]])   # backward edges
                    ]).T
                    
                    edge_index_list.append(edges)
                    edge_attr_list.append(np.ones(edges.shape[0]))
                    
                    # Clear memory
                    del idx1, idx2, mask, edges
                    gc.collect()
            
            # Process feature similarities in small batches for this department
            if len(emp_indices) > 0:
                batch_size = min(100, len(emp_indices))
                for i in range(0, len(emp_indices), batch_size):
                    batch_end = min(i + batch_size, len(emp_indices))
                    batch_indices = emp_indices[i:batch_end]
                    
                    # Only compute similarities with employees not in this department
                    other_indices = np.setdiff1d(np.arange(len(employees)), emp_indices)
                    
                    # Compute similarities
                    batch_features = self.emp_features[batch_indices]
                    other_features = self.emp_features[other_indices]
                    
                    # Normalize features
                    batch_norms = np.linalg.norm(batch_features, axis=1)[:, np.newaxis]
                    other_norms = np.linalg.norm(other_features, axis=1)
                    
                    # Compute similarities
                    similarities = np.dot(batch_features / batch_norms, 
                                       (other_features / other_norms[:, np.newaxis]).T)
                    
                    # Find high similarity pairs
                    high_sim_indices = np.where(similarities > similarity_threshold)
                    
                    if len(high_sim_indices[0]) > 0:
                        # Convert to actual indices
                        src_indices = batch_indices[high_sim_indices[0]]
                        dst_indices = other_indices[high_sim_indices[1]]
                        sim_scores = similarities[high_sim_indices]
                        
                        # Add edges in both directions
                        new_edges = np.vstack([
                            np.stack([src_indices, dst_indices]),
                            np.stack([dst_indices, src_indices])
                        ]).T
                        
                        edge_index_list.append(new_edges)
                        edge_attr_list.append(np.repeat(sim_scores, 2))
                    
                    # Clear memory
                    del batch_features, other_features, similarities
                    gc.collect()
            
            # Periodically combine edges to save memory
            if dept_idx % 3 == 0 and edge_index_list:
                edge_index = np.vstack(edge_index_list)
                edge_attr = np.concatenate(edge_attr_list)
                edge_index_list = [edge_index]
                edge_attr_list = [edge_attr]
                gc.collect()
        
        # Final combination of edges
        print("\nFinalizing edges...")
        edge_index = np.vstack(edge_index_list)
        edge_attr = np.concatenate(edge_attr_list)
        del edge_index_list, edge_attr_list
        gc.collect()
        
        print(f"\nTotal edges created: {len(edge_index)}")
        
        # Convert to PyG format
        print("\nCreating PyG Data object...")
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        x = torch.from_numpy(self.emp_features).float()
        
        # Labels (1 = leaver)
        y = self._create_attrition_labels(employees, dept_emp, cutoff_date)
        
        # Create train/val/test masks
        print("\nCreating data splits...")
        num_nodes = len(employees)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        # Random split (60/20/20)
        indices = torch.randperm(num_nodes)
        train_size = int(0.6 * num_nodes)
        val_size = int(0.2 * num_nodes)
        
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size+val_size]] = True
        test_mask[indices[train_size+val_size:]] = True
        
        data = Data(
            x=x, 
            edge_index=edge_index, 
            edge_attr=edge_attr, 
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask
        )
        data.num_classes = 2
        
        print("\nGraph creation complete!")
        print(f"Number of nodes: {data.num_nodes}")
        print(f"Number of edges: {data.num_edges}")
        print(f"Feature dimension: {data.num_features}")
        
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
        print("\nCreating heterogeneous graph...")
        
        # Build features
        self._prepare_features(employees, departments, dept_emp, titles, salaries, cutoff_date)
        
        # Create HeteroData object
        data = HeteroData()
        
        # Add node features
        data['employee'].x = torch.from_numpy(self.emp_features).float()
        data['department'].x = torch.from_numpy(self.dept_features).float()
        data['title'].x = torch.from_numpy(self.title_features).float()
        
        # Add edges
        print("\nCreating edges...")
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
        
        # Add train/val/test masks
        print("\nCreating data splits...")
        num_nodes = len(employees)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        # Random split (60/20/20)
        indices = torch.randperm(num_nodes)
        train_size = int(0.6 * num_nodes)
        val_size = int(0.2 * num_nodes)
        
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size+val_size]] = True
        test_mask[indices[train_size+val_size:]] = True
        
        data['employee'].train_mask = train_mask
        data['employee'].val_mask = val_mask
        data['employee'].test_mask = test_mask
        
        print("\nHeterogeneous graph creation complete!")
        print(f"Number of employee nodes: {data['employee'].num_nodes}")
        print(f"Number of department nodes: {data['department'].num_nodes}")
        print(f"Number of title nodes: {data['title'].num_nodes}")
        
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
        """Create temporal graph snapshots"""
        print("\nCreating temporal graph snapshots...")
        graphs = []
        
        # Convert prediction horizon to days
        if prediction_horizon.endswith('M'):
            days = int(prediction_horizon[:-1]) * 30
        elif prediction_horizon.endswith('Y'):
            days = int(prediction_horizon[:-1]) * 365
        else:
            days = int(prediction_horizon)
        
        for t in time_windows:
            print(f"\nProcessing snapshot for {t}...")
            # Create graph snapshot at time t
            snapshot_date = self._safe_to_datetime(t)
            label_date = snapshot_date + timedelta(days=days)
            
            # Filter data to snapshot date
            snapshot_dept_emp = dept_emp[dept_emp['to_date'].apply(self._safe_to_datetime) >= snapshot_date]
            snapshot_titles = titles[titles['to_date'].apply(self._safe_to_datetime) >= snapshot_date]
            snapshot_salaries = salaries[salaries['to_date'].apply(self._safe_to_datetime) >= snapshot_date]
            
            # Build graph
            snapshot = self.create_heterogeneous_graph(
                employees, departments, snapshot_dept_emp,
                snapshot_titles, snapshot_salaries,
                cutoff_date=label_date.strftime('%Y-%m-%d')
            )
            graphs.append(snapshot)
            gc.collect()
        
        print(f"\nCreated {len(graphs)} temporal snapshots!")
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
    ) -> Data:
        """Create hierarchical graph with department and employee levels"""
        print("\nCreating hierarchical graph...")
        
        # Build features
        self._prepare_features(employees, departments, dept_emp, titles, salaries, cutoff_date)
        
        # Create a single graph with hierarchical structure
        data = Data()
        
        # Node features
        data.x = torch.from_numpy(self.emp_features).float()
        data.dept_x = torch.from_numpy(self.dept_features).float()
        
        # Edge indices
        print("\nCreating edges...")
        # Employee-employee edges (based on department similarity)
        emp_emp_edges = self._create_emp_emp_edges(employees, dept_emp)
        data.edge_index = emp_emp_edges
        
        # Department-department edges
        dept_dept_edges = self._create_dept_dept_edges(departments, dept_emp)
        data.dept_edge_index = dept_dept_edges
        
        # Employee-department edges for hierarchical connections
        emp_dept_edges = self._create_emp_dept_edges(employees, dept_emp)
        data.hierarchy_edge_index = emp_dept_edges
        
        # Department indices for each employee
        data.dept_idx = self._create_dept_indices(employees, dept_emp, departments)
        
        # Labels and masks
        data.y = self._create_attrition_labels(employees, dept_emp, cutoff_date)
        
        # Create train/val/test masks
        print("\nCreating data splits...")
        num_nodes = len(employees)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        # Random split (60/20/20)
        indices = torch.randperm(num_nodes)
        train_size = int(0.6 * num_nodes)
        val_size = int(0.2 * num_nodes)
        
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size+val_size]] = True
        test_mask[indices[train_size+val_size:]] = True
        
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        
        print("\nHierarchical graph creation complete!")
        print(f"Number of employee nodes: {len(employees)}")
        print(f"Number of department nodes: {len(departments)}")
        
        return data

    def create_knowledge_graph(
        self,
        employees,
        departments,
        dept_emp,
        titles,
        salaries,
        cutoff_date: str = "2002-12-31",
        skills=None  # Optional skills data
    ) -> HeteroData:
        """Create knowledge graph with rich semantic relationships"""
        print("\nCreating knowledge graph...")
        
        # Build features
        self._prepare_features(employees, departments, dept_emp, titles, salaries, cutoff_date)
        
        data = HeteroData()
        
        # Node features
        data['employee'].x = torch.from_numpy(self.emp_features).float()
        data['department'].x = torch.from_numpy(self.dept_features).float()
        data['title'].x = torch.from_numpy(self.title_features).float()
        
        if skills is not None:
            data['skill'].x = torch.from_numpy(self._prepare_skill_features(skills)).float()
        
        # Add edges
        print("\nCreating edges...")
        data['employee', 'works_in', 'department'].edge_index = self._create_emp_dept_edges(employees, dept_emp)
        data['employee', 'has_role', 'title'].edge_index = self._create_emp_title_edges(employees, titles)
        
        if skills is not None:
            data['employee', 'has_skill', 'skill'].edge_index = self._create_emp_skill_edges(employees, skills)
            data['title', 'requires_skill', 'skill'].edge_index = self._create_title_skill_edges(titles, skills)
        
        # Labels
        data['employee'].y = self._create_attrition_labels(employees, dept_emp, cutoff_date)
        
        # Add train/val/test masks
        print("\nCreating data splits...")
        num_nodes = len(employees)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        # Random split (60/20/20)
        indices = torch.randperm(num_nodes)
        train_size = int(0.6 * num_nodes)
        val_size = int(0.2 * num_nodes)
        
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size+val_size]] = True
        test_mask[indices[train_size+val_size:]] = True
        
        data['employee'].train_mask = train_mask
        data['employee'].val_mask = val_mask
        data['employee'].test_mask = test_mask
        
        print("\nKnowledge graph creation complete!")
        print(f"Number of employee nodes: {data['employee'].num_nodes}")
        print(f"Number of department nodes: {data['department'].num_nodes}")
        print(f"Number of title nodes: {data['title'].num_nodes}")
        if skills is not None:
            print(f"Number of skill nodes: {data['skill'].num_nodes}")
        
        # Add metadata
        data.num_node_types = 3 if skills is None else 4
        data.num_edge_types = 2 if skills is None else 4
        data.num_classes = 2
        data.node_types = ['employee', 'department', 'title'] + (['skill'] if skills is not None else [])
        data.edge_types = [
            ('employee', 'works_in', 'department'),
            ('employee', 'has_role', 'title')
        ]
        if skills is not None:
            data.edge_types.extend([
                ('employee', 'has_skill', 'skill'),
                ('title', 'requires_skill', 'skill')
            ])
        
        return data

    def _create_emp_dept_edges(self, employees, dept_emp):
        """Create employee-department edges"""
        emp_id_col = "emp_no" if "emp_no" in employees.columns else "id"
        dept_id_col = "dept_no" if "dept_no" in dept_emp.columns else "id"
        
        # Get latest department assignments
        latest_dept = _latest_by(
            dept_emp,
            by_cols=[emp_id_col],
            sort_cols=['to_date'],
            keep_cols=[emp_id_col, dept_id_col]
        )
        
        # Create edge index
        edge_index = []
        for _, row in latest_dept.iterrows():
            emp_idx = employees[employees[emp_id_col] == row[emp_id_col]].index[0]
            dept_idx = int(row[dept_id_col].replace('d', ''))  # Handle 'd001' format
            edge_index.append([emp_idx, dept_idx])
        
        return torch.tensor(edge_index, dtype=torch.long).t()
    
    def _create_emp_title_edges(self, employees, titles):
        """Create employee-title edges"""
        emp_id_col = "emp_no" if "emp_no" in employees.columns else "id"
        
        # Get latest titles
        latest_titles = _latest_by(
            titles,
            by_cols=[emp_id_col],
            sort_cols=['to_date'],
            keep_cols=[emp_id_col, 'title']
        )
        
        # Create edge index
        edge_index = []
        title_to_idx = {title: idx for idx, title in enumerate(latest_titles['title'].unique())}
        
        for _, row in latest_titles.iterrows():
            emp_idx = employees[employees[emp_id_col] == row[emp_id_col]].index[0]
            title_idx = title_to_idx[row['title']]
            edge_index.append([emp_idx, title_idx])
        
        return torch.tensor(edge_index, dtype=torch.long).t()
    
    def _create_dept_title_edges(self, dept_emp, titles):
        """Create department-title edges"""
        # This is a simplified implementation
        # In practice, you'd want to track which titles exist in which departments
        return torch.tensor([[], []], dtype=torch.long)
    
    def _create_dept_dept_edges(self, departments, dept_emp):
        """Create department-department edges based on employee transfers"""
        dept_pairs = dept_emp.groupby('emp_no')['dept_no'].apply(list)
        edge_index = []
        for depts in dept_pairs:
            if len(depts) > 1:
                for i in range(len(depts)-1):
                    dept_i = int(depts[i].replace('d', ''))
                    dept_j = int(depts[i+1].replace('d', ''))
                    edge_index.append([dept_i, dept_j])
                    edge_index.append([dept_j, dept_i])  # Make it bidirectional
        
        if not edge_index:  # If no transfers found
            return torch.tensor([[], []], dtype=torch.long)
        return torch.tensor(edge_index, dtype=torch.long).t()
    
    def _create_dept_indices(self, employees, dept_emp, departments):
        """Create department indices for each employee"""
        emp_id_col = "emp_no" if "emp_no" in employees.columns else "id"
        dept_id_col = "dept_no" if "dept_no" in departments.columns else "id"
        
        latest_dept = _latest_by(
            dept_emp,
            by_cols=[emp_id_col],
            sort_cols=['to_date'],
            keep_cols=[emp_id_col, dept_id_col]
        )
        
        dept_idx = torch.zeros(len(employees), dtype=torch.long)
        for _, row in latest_dept.iterrows():
            emp_idx = employees[employees[emp_id_col] == row[emp_id_col]].index[0]
            dept_idx[emp_idx] = int(row[dept_id_col].replace('d', ''))
        
        return dept_idx
    
    def _create_attrition_labels(self, employees, dept_emp, cutoff_date):
        """Create binary attrition labels"""
        cutoff = self._safe_to_datetime(cutoff_date)
        latest_dept = _latest_by(
            dept_emp,
            by_cols=['emp_no'],
            sort_cols=['to_date'],
            keep_cols=['emp_no', 'to_date']
        )
        
        # 1 = leaver (left before cutoff)
        labels = torch.zeros(len(employees), dtype=torch.long)
        
        # Handle the special case of '9999-01-01' which indicates current employees
        for idx, row in latest_dept.iterrows():
            if pd.notna(row['to_date']):
                to_date_str = str(row['to_date'])
                # Current employee (still employed)
                if to_date_str == CURRENT_DATE:
                    continue
                # Left before cutoff
                if self._safe_to_datetime(to_date_str) < cutoff:
                    emp_idx = employees[employees['emp_no'] == row['emp_no']].index[0]
                    labels[emp_idx] = 1
                
        return labels
    
    def _create_emp_emp_edges(self, employees, dept_emp):
        """Create employee-employee edges based on department co-membership"""
        emp_id_col = "emp_no" if "emp_no" in employees.columns else "id"
        dept_id_col = "dept_no" if "dept_no" in dept_emp.columns else "id"
        
        # Get latest department assignments
        latest_dept = _latest_by(
            dept_emp,
            by_cols=[emp_id_col],
            sort_cols=['to_date'],
            keep_cols=[emp_id_col, dept_id_col]
        )
        
        # Group employees by department
        dept_groups = latest_dept.groupby(dept_id_col)[emp_id_col].apply(list)
        edge_index = []
        
        # Create edges between all employees in the same department
        for dept_employees in dept_groups:
            if len(dept_employees) > 1:
                emp_indices = []
                for emp_no in dept_employees:
                    try:
                        idx = employees[employees[emp_id_col] == emp_no].index[0]
                        emp_indices.append(idx)
                    except IndexError:
                        continue
                
                if len(emp_indices) > 1:
                    # Create edges between all pairs (both directions)
                    for i in range(len(emp_indices)):
                        for j in range(i + 1, len(emp_indices)):
                            edge_index.append([emp_indices[i], emp_indices[j]])
                            edge_index.append([emp_indices[j], emp_indices[i]])
        
        if not edge_index:
            return torch.tensor([[], []], dtype=torch.long)
        return torch.tensor(edge_index, dtype=torch.long).t()

    def _prepare_skill_features(self, skills):
        """Prepare skill features"""
        # Simple implementation - can be enhanced
        if skills is None:
            return np.zeros((1, 5), dtype=np.float32)
        return np.eye(len(skills), dtype=np.float32)