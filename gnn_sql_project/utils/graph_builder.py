import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

# ---------------------------
# helpers
# ---------------------------

def _to_datetime(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def _latest_by(df, by_cols, sort_cols, keep_cols):
    """
    Return latest row per 'by_cols' using max(sort_cols) ordering.
    """
    if not set(by_cols).issubset(df.columns):
        return pd.DataFrame(columns=keep_cols)
    df2 = df.dropna(subset=by_cols).copy()
    df2 = df2.sort_values(by=by_cols + sort_cols)
    last = df2.groupby(by_cols, as_index=False).tail(1)
    return last[keep_cols].copy()

def _standardize(df, numeric_cols):
    df = df.copy()
    for c in numeric_cols:
        m = df[c].mean()
        s = df[c].std()
        if not np.isfinite(m):
            m = 0.0
        if not np.isfinite(s) or s < 1e-6:
            s = 1.0
        df[c] = (df[c] - m) / s
    return df

# ---------------------------
# main builder
# ---------------------------

def create_graph(employees, departments, dept_emp, dept_manager, titles, salaries):
    """
    Build a bipartite employee–department graph.

    Nodes:
      - first |E| nodes are employees
      - next  |D| nodes are departments

    Edges:
      - undirected edges between employee i and department j for each row in dept_emp

    Labels:
      - employee node label = index of their latest department (0..K-1)
      - department nodes carry a dummy label 0 and are masked out by your main script

    Features (all float32):
      Employees:
        [age_z, tenure_years_z, current_salary_z, salary_growth_z, title_code_z, <9 dept one-hot>]
      Departments:
        [0,0,0,0,0, <9 dept identity one-hot>]
    """
    # Column picks (robust to your schema)
    emp_id_col = "emp_no" if "emp_no" in employees.columns else "id"
    dept_id_col = "dept_no" if "dept_no" in departments.columns else "id"

    print(f"[graph_builder] Picked columns → emp_id:{emp_id_col}, dept_id:{dept_id_col}")

    # defensive copies
    employees = employees.copy()
    departments = departments.copy()
    dept_emp = dept_emp.copy()
    titles = titles.copy()
    salaries = salaries.copy()
    dept_manager = dept_manager.copy()

    # types
    if employees[emp_id_col].dtype != "int64":
        employees[emp_id_col] = pd.to_numeric(employees[emp_id_col], errors="coerce").astype("Int64").fillna(-1).astype("int64")
    if dept_emp[emp_id_col].dtype != "int64":
        dept_emp[emp_id_col] = pd.to_numeric(dept_emp[emp_id_col], errors="coerce").astype("Int64").fillna(-1).astype("int64")
    if salaries.get("salary") is not None and salaries[emp_id_col].dtype != "int64":
        salaries[emp_id_col] = pd.to_numeric(salaries[emp_id_col], errors="coerce").astype("Int64").fillna(-1).astype("int64")
    if titles.get(emp_id_col) is not None and titles[emp_id_col].dtype != "int64":
        titles[emp_id_col] = pd.to_numeric(titles[emp_id_col], errors="coerce").astype("Int64").fillna(-1).astype("int64")

    # string department ids
    departments[dept_id_col] = departments[dept_id_col].astype(str)
    dept_emp[dept_id_col] = dept_emp[dept_id_col].astype(str)
    if dept_manager.shape[0] > 0 and dept_id_col in dept_manager.columns:
        dept_manager[dept_id_col] = dept_manager[dept_id_col].astype(str)

    # dates
    _to_datetime(employees, ["birth_date", "hire_date"])
    _to_datetime(dept_emp, ["from_date", "to_date"])
    _to_datetime(salaries, ["from_date", "to_date"])
    _to_datetime(titles, ["from_date", "to_date"])
    _to_datetime(dept_manager, ["from_date", "to_date"])

    # ---- node index maps ----
    emp_ids = employees[emp_id_col].astype("int64").tolist()
    dept_ids = departments[dept_id_col].astype(str).tolist()
    emp_to_idx = {eid: i for i, eid in enumerate(emp_ids)}
    dept_to_idx = {did: i for i, did in enumerate(dept_ids)}
    num_emp = len(emp_ids)
    num_dept = len(dept_ids)

    # ---- features: employees ----
    today = pd.to_datetime("1999-01-01")  # consistent with dataset timeframe
    employees["age_years"] = ((today - employees["birth_date"]).dt.days / 365.25).clip(lower=0)
    employees["tenure_years"] = ((today - employees["hire_date"]).dt.days / 365.25).clip(lower=0)

    # latest department per employee (also used for labels & one-hot)
    d_latest = _latest_by(
        dept_emp,
        by_cols=[emp_id_col],
        sort_cols=["to_date"],
        keep_cols=[emp_id_col, dept_id_col, "to_date"]
    )
    d_latest = d_latest.rename(columns={dept_id_col: "dept_latest"})

    # latest salary per employee
    s_latest = _latest_by(
        salaries,
        by_cols=[emp_id_col],
        sort_cols=["to_date"],
        keep_cols=[emp_id_col, "salary", "to_date"]
    ).rename(columns={"salary": "curr_salary", "to_date": "sal_to"})

    # salary growth per employee: (last - first) / years
    if salaries.shape[0] > 0:
        sal = salaries[[emp_id_col, "salary", "from_date"]].dropna(subset=[emp_id_col, "salary"])
        sal = sal.sort_values([emp_id_col, "from_date"])
        first = sal.groupby(emp_id_col, as_index=False).head(1).rename(columns={"salary": "sal_first", "from_date": "sal_from_first"})
        last = sal.groupby(emp_id_col, as_index=False).tail(1).rename(columns={"salary": "sal_last", "from_date": "sal_from_last"})
        sg = pd.merge(first[[emp_id_col, "sal_first", "sal_from_first"]],
                      last[[emp_id_col, "sal_last", "sal_from_last"]],
                      on=emp_id_col, how="outer")
        # duration in years (>= ~0 to avoid div by zero)
        dur_years = (sg["sal_from_last"] - sg["sal_from_first"]).dt.days / 365.25
        dur_years = dur_years.where(dur_years.abs() > 1e-9, other=1.0)
        sg["salary_growth"] = (sg["sal_last"] - sg["sal_first"]) / dur_years
        sg = sg[[emp_id_col, "salary_growth"]]
    else:
        sg = pd.DataFrame({emp_id_col: emp_ids, "salary_growth": 0.0})

    # latest title (encode to category code)
    if titles.shape[0] > 0 and "title" in titles.columns:
        t_latest = _latest_by(
            titles,
            by_cols=[emp_id_col],
            sort_cols=["to_date"],
            keep_cols=[emp_id_col, "title"]
        )
        t_latest["title_code"] = t_latest["title"].astype("category").cat.codes.astype("int64")
        t_latest = t_latest[[emp_id_col, "title_code"]]
    else:
        t_latest = pd.DataFrame({emp_id_col: emp_ids, "title_code": 0})

    # assemble employee features
    emp_feat = employees[[emp_id_col, "age_years", "tenure_years"]]
    emp_feat = emp_feat.merge(s_latest[[emp_id_col, "curr_salary"]], on=emp_id_col, how="left")
    emp_feat = emp_feat.merge(sg, on=emp_id_col, how="left")
    emp_feat = emp_feat.merge(t_latest, on=emp_id_col, how="left")
    emp_feat = emp_feat.merge(d_latest[[emp_id_col, "dept_latest"]], on=emp_id_col, how="left")

    # fill numeric NaNs
    for c in ["age_years", "tenure_years", "curr_salary", "salary_growth", "title_code"]:
        if c not in emp_feat.columns:
            emp_feat[c] = 0.0
    emp_feat[["age_years", "tenure_years", "curr_salary", "salary_growth"]] = \
        emp_feat[["age_years", "tenure_years", "curr_salary", "salary_growth"]].fillna(0.0)
    emp_feat["title_code"] = emp_feat["title_code"].fillna(0).astype("int64")

    # department one-hot (9)
    dept_list = sorted(departments[dept_id_col].astype(str).unique().tolist())
    dept_to_class = {d: i for i, d in enumerate(dept_list)}
    K = len(dept_list)

    # create one-hot for employees based on latest dept
    emp_feat["dept_latest"] = emp_feat["dept_latest"].astype(str)
    emp_dept_idx = emp_feat["dept_latest"].map(dept_to_class).fillna(-1).astype("int64")
    emp_dept_onehot = np.zeros((emp_feat.shape[0], K), dtype=np.float32)
    valid_mask = emp_dept_idx.values >= 0
    emp_dept_onehot[np.where(valid_mask)[0], emp_dept_idx.values[valid_mask]] = 1.0

    # standardize numeric block
    numeric_cols = ["age_years", "tenure_years", "curr_salary", "salary_growth", "title_code"]
    emp_feat_std = _standardize(emp_feat[numeric_cols].copy(), numeric_cols)

    # final employee features matrix: numeric(5) + dept_onehot(9) = 14 dims
    emp_features = np.hstack([emp_feat_std.values.astype(np.float32), emp_dept_onehot])

    # ---- features: departments ----
    # departments get zeros for numeric(5) + identity one-hot(9)
    dept_identity = np.eye(K, dtype=np.float32)
    dept_numeric_zeros = np.zeros((num_dept, len(numeric_cols)), dtype=np.float32)
    dept_features = np.hstack([dept_numeric_zeros, dept_identity])

    # ---- concat node features ----
    x = torch.from_numpy(np.vstack([emp_features, dept_features]).astype(np.float32))
    D = x.shape[1]
    print(f"[graph_builder] Feature width D={D} (emp dims={emp_features.shape[1]}, dept one-hot={K})")

    # ---- edges (employee<->department from dept_emp) ----
    # Use all historical rows to strengthen structure
    e = dept_emp[[emp_id_col, dept_id_col]].dropna()
    e = e[(e[emp_id_col].isin(emp_to_idx)) & (e[dept_id_col].isin(dept_to_idx))]

    src = []
    dst = []
    for row in e.itertuples(index=False):
        ei = emp_to_idx[int(getattr(row, emp_id_col))]
        dj = dept_to_idx[str(getattr(row, dept_id_col))] + num_emp  # offset dept index
        src.append(ei); dst.append(dj)
        src.append(dj); dst.append(ei)  # undirected (symmetrize)

    if len(src) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)

    print(f"[graph_builder] Edge count: {edge_index.shape[1]} | Classes: {K}")

    # ---- labels (employees only) = latest department class ----
    y_emp = torch.full((num_emp,), -1, dtype=torch.long)
    # use d_latest for labels
    dlab = d_latest[[emp_id_col, "dept_latest"]].copy()
    dlab["cls"] = dlab["dept_latest"].map(dept_to_class).astype("Int64").fillna(-1).astype(int)

    matched = 0
    for r in dlab.itertuples(index=False):
        eid = int(getattr(r, emp_id_col))
        cls = int(getattr(r, "cls"))
        if eid in emp_to_idx and cls >= 0:
            y_emp[emp_to_idx[eid]] = cls
            matched += 1

    # fallback: any -1 gets majority class
    if (y_emp == -1).any():
        vals = y_emp[y_emp >= 0].tolist()
        majority = max(set(vals), key=vals.count) if vals else 0
        y_emp[y_emp == -1] = majority

    # dept labels are dummy zeros
    y_dept = torch.zeros(num_dept, dtype=torch.long)
    y = torch.cat([y_emp, y_dept], dim=0)

    # meta
    used_classes = int(torch.unique(y_emp).numel())
    print(f"[graph_builder] Labelled employees: {matched}/{num_emp} | #classes in use: {used_classes}")

    data = Data(x=x, edge_index=edge_index, y=y)
    data.num_employees = num_emp
    data.num_departments = num_dept
    data.num_classes = max(2, K)

    return data
