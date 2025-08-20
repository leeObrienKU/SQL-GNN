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

def create_graph(
    employees,
    departments,
    dept_emp,
    dept_manager,
    titles,
    salaries,
    task: str = "dept",          # "dept" (old) or "attrition" (new)
    cutoff_date: str | None = None,   # e.g. "2000-01-01" (only for attrition)
    use_all_history_edges: bool = True
):
    """
    Build a bipartite employee–department graph.

    Nodes:
      - first |E| nodes are employees
      - next  |D| nodes are departments

    Edges:
      - undirected edges between employee i and department j for each row in dept_emp
        (optionally all historical assignments)

    Labels (by task):
      - task="dept"      → employee label = index of latest department (0..K-1)
      - task="attrition" → employee label ∈ {0,1}, 1 = leaver
            default rule: if latest to_date < 9999-01-01 then leaver=1, else 0
            if cutoff_date is set: if latest to_date < cutoff_date then 1, else 0

    Features (float32):
      Employees:
        numeric(5): [age_z, tenure_years_z, current_salary_z, salary_growth_z, title_code_z]
        + dept one-hot (K)
      Departments:
        zeros(5) + identity one-hot (K)
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
    for df, col in [(employees, emp_id_col), (dept_emp, emp_id_col), (salaries, emp_id_col), (titles, emp_id_col)]:
        if col in df.columns and df[col].dtype != "int64":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64").fillna(-1).astype("int64")

    # string department ids
    if dept_id_col in departments.columns:
        departments[dept_id_col] = departments[dept_id_col].astype(str)
    if dept_id_col in dept_emp.columns:
        dept_emp[dept_id_col] = dept_emp[dept_id_col].astype(str)
    if dept_id_col in dept_manager.columns:
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

    # ---- reference date for features ----
    # Keep in dataset timeframe; if cutoff provided, use that as reference
    sentinel = pd.Timestamp("9999-01-01")
    ref_date = pd.to_datetime(cutoff_date) if cutoff_date else pd.Timestamp("1999-01-01")

    # ---- features: employees ----
    employees["age_years"] = ((ref_date - employees["birth_date"]).dt.days / 365.25).clip(lower=0)
    employees["tenure_years"] = ((ref_date - employees["hire_date"]).dt.days / 365.25).clip(lower=0)

    # latest department per employee (also used for labels & one-hot)
    d_latest = _latest_by(
        dept_emp,
        by_cols=[emp_id_col],
        sort_cols=["to_date"],
        keep_cols=[emp_id_col, dept_id_col, "to_date"]
    ).rename(columns={dept_id_col: "dept_latest"})

    # latest salary per employee (<= ref_date if available)
    s = salaries.copy()
    if "to_date" in s.columns:
        s = s[(s["to_date"].isna()) | (s["to_date"] <= ref_date)]
    s_latest = _latest_by(
        s,
        by_cols=[emp_id_col],
        sort_cols=["to_date"] if "to_date" in s.columns else ["from_date"],
        keep_cols=[emp_id_col, "salary", "to_date"] if "to_date" in s.columns else [emp_id_col, "salary", "from_date"]
    ).rename(columns={"salary": "curr_salary"})

    # salary growth per employee: (last - first) / years using rows <= ref_date
    if salaries.shape[0] > 0:
        sal = salaries[[emp_id_col, "salary", "from_date"]].dropna(subset=[emp_id_col, "salary"]).copy()
        sal = sal[sal["from_date"] <= ref_date]
        sal = sal.sort_values([emp_id_col, "from_date"])
        first = sal.groupby(emp_id_col, as_index=False).head(1).rename(columns={"salary": "sal_first", "from_date": "sal_from_first"})
        last = sal.groupby(emp_id_col, as_index=False).tail(1).rename(columns={"salary": "sal_last", "from_date": "sal_from_last"})
        sg = pd.merge(first[[emp_id_col, "sal_first", "sal_from_first"]],
                      last[[emp_id_col, "sal_last", "sal_from_last"]],
                      on=emp_id_col, how="outer")
        dur_years = (sg["sal_from_last"] - sg["sal_from_first"]).dt.days / 365.25
        dur_years = dur_years.where(dur_years.abs() > 1e-9, other=1.0)
        sg["salary_growth"] = (sg["sal_last"] - sg["sal_first"]) / dur_years
        sg = sg[[emp_id_col, "salary_growth"]]
    else:
        sg = pd.DataFrame({emp_id_col: emp_ids, "salary_growth": 0.0})

    # latest title (encode to category code, up to ref_date)
    if titles.shape[0] > 0 and "title" in titles.columns:
        t = titles.copy()
        if "to_date" in t.columns:
            t = t[(t["to_date"].isna()) | (t["to_date"] <= ref_date)]
        t_latest = _latest_by(
            t,
            by_cols=[emp_id_col],
            sort_cols=["to_date"] if "to_date" in t.columns else ["from_date"],
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

    # department one-hot (K)
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

    # final employee features matrix: numeric(5) + dept_onehot(K)
    emp_features = np.hstack([emp_feat_std.values.astype(np.float32), emp_dept_onehot])

    # ---- features: departments ----
    dept_identity = np.eye(K, dtype=np.float32)
    dept_numeric_zeros = np.zeros((num_dept, len(numeric_cols)), dtype=np.float32)
    dept_features = np.hstack([dept_numeric_zeros, dept_identity])

    # ---- concat node features ----
    x = torch.from_numpy(np.vstack([emp_features, dept_features]).astype(np.float32))
    D = x.shape[1]
    print(f"[graph_builder] Feature width D={D} (emp dims={emp_features.shape[1]}, dept one-hot={K})")

    # ---- edges (employee<->department from dept_emp) ----
    e = dept_emp[[emp_id_col, dept_id_col, "to_date"]].dropna(subset=[emp_id_col, dept_id_col]).copy() \
        if {"to_date"}.issubset(dept_emp.columns) else dept_emp[[emp_id_col, dept_id_col]].dropna().copy()

    if not use_all_history_edges and "to_date" in e.columns:
        # keep only edges that are active as of ref_date
        # (active if to_date is NaT or >= ref_date)
        e = e[(e["to_date"].isna()) | (e["to_date"] >= ref_date)]

    e = e[(e[emp_id_col].isin(emp_to_idx)) & (e[dept_id_col].isin(dept_to_idx))]

    src, dst = [], []
    for row in e.itertuples(index=False):
        ei = emp_to_idx[int(getattr(row, emp_id_col))]
        dj = dept_to_idx[str(getattr(row, dept_id_col))] + num_emp  # offset dept index
        src.append(ei); dst.append(dj)
        src.append(dj); dst.append(ei)  # undirected (symmetrize)

    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.empty((2, 0), dtype=torch.long)
    print(f"[graph_builder] Edge count: {edge_index.shape[1]} | Classes (depts): {K}")

    # ---- labels ----
    if task == "attrition":
        # 1 = leaver
        dlab = d_latest[[emp_id_col, "to_date"]].copy()
        cutoff = pd.to_datetime(cutoff_date) if cutoff_date else None

        if cutoff is None:
            # classic sentinel logic
            dlab["leaver"] = ((~dlab["to_date"].isna()) & (dlab["to_date"] < sentinel)).astype(int)
        else:
            dlab["leaver"] = ((~dlab["to_date"].isna()) & (dlab["to_date"] < cutoff)).astype(int)

        y_emp = torch.zeros(num_emp, dtype=torch.long)
        matched = 0
        for r in dlab.itertuples(index=False):
            eid = int(getattr(r, emp_id_col))
            lab = int(getattr(r, "leaver"))
            if eid in emp_to_idx:
                y_emp[emp_to_idx[eid]] = lab
                matched += 1

        y_dept = torch.zeros(num_dept, dtype=torch.long)  # keep depts = 0 (they'll be masked later)
        y = torch.cat([y_emp, y_dept], dim=0)
        used_classes = int(torch.unique(y_emp).numel())
        print(f"[graph_builder] Labelled employees (attrition): {matched}/{num_emp} | #classes in use: {used_classes}")
        num_classes = 2

    else:
        # task == "dept" (backward compatible)
        y_emp = torch.full((num_emp,), -1, dtype=torch.long)
        dlab = d_latest[[emp_id_col, "dept_latest"]].copy()
        dept_to_class = {d: i for i, d in enumerate(dept_list)}
        dlab["cls"] = dlab["dept_latest"].map(dept_to_class).astype("Int64").fillna(-1).astype(int)

        matched = 0
        for r in dlab.itertuples(index=False):
            eid = int(getattr(r, emp_id_col))
            cls = int(getattr(r, "cls"))
            if eid in emp_to_idx and cls >= 0:
                y_emp[emp_to_idx[eid]] = cls
                matched += 1

        if (y_emp == -1).any():
            vals = y_emp[y_emp >= 0].tolist()
            majority = max(set(vals), key=vals.count) if vals else 0
            y_emp[y_emp == -1] = majority

        y_dept = torch.zeros(num_dept, dtype=torch.long)
        y = torch.cat([y_emp, y_dept], dim=0)
        used_classes = int(torch.unique(y_emp).numel())
        print(f"[graph_builder] Labelled employees (dept): {matched}/{num_emp} | #classes in use: {used_classes}")
        num_classes = max(2, K)

    # meta
    data = Data(x=x, edge_index=edge_index, y=y)
    data.num_employees = num_emp
    data.num_departments = num_dept
    data.num_classes = num_classes
    data.task = task
    data.ref_date = str(ref_date.date())

    return data
