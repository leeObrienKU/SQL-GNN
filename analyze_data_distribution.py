import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import psycopg2
from datetime import datetime
import os
from utils.eda_visualizer import EDAVisualizer

def connect_to_db():
    """Establish database connection"""
    try:
        # Try different connection options
        connection_params = [
            {
                'dbname': 'empdb',
                'user': 'postgres',
                'host': '/var/run/postgresql'
            },
            {
                'dbname': 'empdb',
                'user': 'postgres',
                'host': 'localhost'
            },
            {
                'dbname': 'empdb',
                'user': 'postgres',
                'host': '127.0.0.1'
            }
        ]
        
        last_error = None
        for params in connection_params:
            try:
                conn = psycopg2.connect(**params)
                print(f"✅ Connected to PostgreSQL at {params['host']}")
                return conn
            except Exception as e:
                last_error = e
                continue
        
        print("❌ Failed to connect to database:")
        print(f"   • Last error: {last_error}")
        print("   • Tried connecting to:")
        for params in connection_params:
            print(f"     - {params['host']}")
        return None
        
    except Exception as e:
        print(f"❌ Unexpected database error: {e}")
        return None

def analyze_temporal_distribution(cur, output_dir):
    """Analyze the temporal distribution of all events in the database"""
    print("\n" + "=" * 80)
    print("📊 TEMPORAL DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    # Create figure for temporal distributions
    plt.figure(figsize=(15, 10))
    
    # 1. Employment periods
    cur.execute("""
        SELECT 
            MIN(from_date) as earliest_start,
            MAX(CASE WHEN to_date = '9999-01-01' THEN CURRENT_DATE ELSE to_date END) as latest_end,
            COUNT(*) as total_records,
            COUNT(DISTINCT employee_id) as unique_employees,
            COUNT(CASE WHEN to_date = '9999-01-01' THEN 1 END) as current_employees
        FROM employees.department_employee
    """)
    emp_range = cur.fetchone()
    
    print("\n📅 Employment Timeline:")
    print(f"  • Data Range: {emp_range[0]} to {emp_range[1]}")
    print(f"  • Total Employment Records: {emp_range[2]:,}")
    print(f"  • Unique Employees: {emp_range[3]:,}")
    print(f"  • Current Employees: {emp_range[4]:,}")
    
    # Get yearly distribution of employment starts and ends
    cur.execute("""
        SELECT 
            EXTRACT(YEAR FROM from_date) as year,
            COUNT(*) as starts,
            COUNT(CASE WHEN to_date != '9999-01-01' THEN 1 END) as ends
        FROM employees.department_employee
        GROUP BY EXTRACT(YEAR FROM from_date)
        ORDER BY year
    """)
    yearly_emp = pd.DataFrame(cur.fetchall(), columns=['year', 'starts', 'ends'])
    
    # Plot employment distribution
    plt.subplot(2, 2, 1)
    plt.plot(yearly_emp['year'], yearly_emp['starts'], label='New Employments')
    plt.plot(yearly_emp['year'], yearly_emp['ends'], label='Terminations')
    plt.title('Employment Events by Year')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    
    # 2. Salary changes
    cur.execute("""
        SELECT 
            EXTRACT(YEAR FROM from_date) as year,
            COUNT(*) as changes,
            AVG(amount) as avg_amount,
            COUNT(DISTINCT employee_id) as employees_affected
        FROM employees.salary
        GROUP BY EXTRACT(YEAR FROM from_date)
        ORDER BY year
    """)
    salary_dist = pd.DataFrame(cur.fetchall(), columns=['year', 'changes', 'avg_amount', 'employees'])
    
    plt.subplot(2, 2, 2)
    plt.plot(salary_dist['year'], salary_dist['avg_amount'])
    plt.title('Average Salary Over Time')
    plt.xlabel('Year')
    plt.ylabel('Average Salary')
    plt.grid(True)
    
    # 3. Department transfers
    cur.execute("""
        WITH transfers AS (
            SELECT 
                employee_id,
                from_date,
                to_date,
                department_id,
                LAG(department_id) OVER (PARTITION BY employee_id ORDER BY from_date) as prev_dept
            FROM employees.department_employee
        )
        SELECT 
            EXTRACT(YEAR FROM from_date) as year,
            COUNT(*) as transfers
        FROM transfers
        WHERE department_id != prev_dept OR prev_dept IS NULL
        GROUP BY EXTRACT(YEAR FROM from_date)
        ORDER BY year
    """)
    transfer_dist = pd.DataFrame(cur.fetchall(), columns=['year', 'transfers'])
    
    plt.subplot(2, 2, 3)
    plt.plot(transfer_dist['year'], transfer_dist['transfers'])
    plt.title('Department Transfers by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Transfers')
    plt.grid(True)
    
    # 4. Title changes
    cur.execute("""
        SELECT 
            EXTRACT(YEAR FROM from_date) as year,
            COUNT(*) as changes,
            COUNT(DISTINCT employee_id) as employees
        FROM employees.title
        GROUP BY EXTRACT(YEAR FROM from_date)
        ORDER BY year
    """)
    title_dist = pd.DataFrame(cur.fetchall(), columns=['year', 'changes', 'employees'])
    
    plt.subplot(2, 2, 4)
    plt.plot(title_dist['year'], title_dist['changes'])
    plt.title('Title Changes by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Changes')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temporal_distribution.png'))
    plt.close()
    
    # Analyze employment duration patterns
    cur.execute("""
        WITH emp_duration AS (
            SELECT 
                employee_id,
                MIN(from_date) as first_date,
                MAX(CASE WHEN to_date = '9999-01-01' THEN CURRENT_DATE ELSE to_date END) as last_date
            FROM employees.department_employee
            GROUP BY employee_id
        )
        SELECT 
            EXTRACT(YEAR FROM AGE(last_date, first_date)) as years_employed,
            COUNT(*) as employee_count
        FROM emp_duration
        GROUP BY EXTRACT(YEAR FROM AGE(last_date, first_date))
        ORDER BY years_employed
    """)
    duration_dist = pd.DataFrame(cur.fetchall(), columns=['years', 'count'])
    
    plt.figure(figsize=(10, 6))
    plt.bar(duration_dist['years'], duration_dist['count'])
    plt.title('Employment Duration Distribution')
    plt.xlabel('Years Employed')
    plt.ylabel('Number of Employees')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'employment_duration.png'))
    plt.close()
    
    print("\n📊 Employment Duration Statistics:")
    # Calculate weighted average
    avg_duration = np.average(duration_dist['years'], weights=duration_dist['count'])
    
    # Calculate weighted median
    cumsum = duration_dist['count'].cumsum()
    total = duration_dist['count'].sum()
    median_idx = duration_dist.index[cumsum >= total/2].min()
    median_duration = duration_dist.loc[median_idx, 'years']
    
    # Get mode (most common)
    mode_duration = duration_dist.loc[duration_dist['count'].idxmax(), 'years']
    
    print(f"  • Average Duration: {avg_duration:.1f} years")
    print(f"  • Median Duration: {median_duration:.1f} years")
    print(f"  • Most Common Duration: {mode_duration:.1f} years")

def analyze_department_patterns(cur, output_dir):
    """Analyze department-specific patterns"""
    print("\n" + "=" * 80)
    print("🏢 DEPARTMENT ANALYSIS")
    print("=" * 80)
    
    # Department size and stability
    cur.execute("""
        WITH dept_stats AS (
            SELECT 
                d.dept_name,
                COUNT(DISTINCT de.employee_id) as total_employees,
                COUNT(DISTINCT CASE WHEN de.to_date = '9999-01-01' THEN de.employee_id END) as current_employees,
                COUNT(DISTINCT CASE WHEN de.to_date != '9999-01-01' THEN de.employee_id END) as former_employees,
                AVG(EXTRACT(YEAR FROM AGE(
                    CASE WHEN de.to_date = '9999-01-01' THEN CURRENT_DATE ELSE de.to_date END,
                    de.from_date
                ))) as avg_tenure
            FROM employees.department d
            JOIN employees.department_employee de ON d.id = de.department_id
            GROUP BY d.id, d.dept_name
        )
        SELECT 
            dept_name,
            total_employees,
            current_employees,
            former_employees,
            ROUND(avg_tenure::numeric, 2) as avg_tenure_years,
            ROUND(former_employees * 100.0 / total_employees, 2) as turnover_rate
        FROM dept_stats
        ORDER BY total_employees DESC
    """)
    dept_stats = pd.DataFrame(cur.fetchall(), 
                            columns=['Department', 'Total', 'Current', 'Former', 'Avg Tenure', 'Turnover Rate'])
    
    print("\nDepartment Statistics:")
    print(dept_stats.to_string(index=False))
    
    # Plot department metrics
    plt.figure(figsize=(15, 10))
    
    # Size comparison
    plt.subplot(2, 2, 1)
    plt.bar(dept_stats['Department'], dept_stats['Current'], label='Current')
    plt.bar(dept_stats['Department'], dept_stats['Former'], 
            bottom=dept_stats['Current'], label='Former')
    plt.title('Department Size Composition')
    plt.xticks(rotation=45)
    plt.legend()
    
    # Turnover rate
    plt.subplot(2, 2, 2)
    plt.bar(dept_stats['Department'], dept_stats['Turnover Rate'])
    plt.title('Department Turnover Rate (%)')
    plt.xticks(rotation=45)
    
    # Average tenure
    plt.subplot(2, 2, 3)
    plt.bar(dept_stats['Department'], dept_stats['Avg Tenure'])
    plt.title('Average Employee Tenure (Years)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'department_metrics.png'))
    plt.close()

def analyze_salary_patterns(cur, output_dir):
    """Analyze salary patterns and growth"""
    print("\n" + "=" * 80)
    print("💰 SALARY ANALYSIS")
    print("=" * 80)
    
    # Overall salary trends
    cur.execute("""
        WITH salary_changes AS (
            SELECT 
                employee_id,
                amount,
                from_date,
                LAG(amount) OVER (PARTITION BY employee_id ORDER BY from_date) as prev_salary,
                ROW_NUMBER() OVER (PARTITION BY employee_id ORDER BY from_date) as salary_change_num
            FROM employees.salary
        )
        SELECT 
            EXTRACT(YEAR FROM from_date) as year,
            AVG(amount) as avg_salary,
            AVG(CASE WHEN prev_salary IS NOT NULL 
                THEN (amount - prev_salary) * 100.0 / prev_salary 
                END) as avg_increase,
            COUNT(DISTINCT employee_id) as employees_with_changes,
            AVG(salary_change_num) as avg_changes_per_employee
        FROM salary_changes
        GROUP BY EXTRACT(YEAR FROM from_date)
        ORDER BY year
    """)
    salary_trends = pd.DataFrame(cur.fetchall(), 
                               columns=['Year', 'Avg Salary', 'Avg Increase %', 
                                      'Employees Changed', 'Avg Changes/Employee'])
    
    print("\nSalary Trend Statistics:")
    print(salary_trends.to_string(index=False))
    
    # Plot salary metrics
    plt.figure(figsize=(15, 10))
    
    # Average salary trend
    plt.subplot(2, 2, 1)
    plt.plot(salary_trends['Year'], salary_trends['Avg Salary'])
    plt.title('Average Salary Over Time')
    plt.grid(True)
    
    # Average increase percentage
    plt.subplot(2, 2, 2)
    plt.plot(salary_trends['Year'], salary_trends['Avg Increase %'])
    plt.title('Average Salary Increase (%)')
    plt.grid(True)
    
    # Number of employees with changes
    plt.subplot(2, 2, 3)
    plt.plot(salary_trends['Year'], salary_trends['Employees Changed'])
    plt.title('Employees with Salary Changes')
    plt.grid(True)
    
    # Average changes per employee
    plt.subplot(2, 2, 4)
    plt.plot(salary_trends['Year'], salary_trends['Avg Changes/Employee'])
    plt.title('Average Salary Changes per Employee')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'salary_metrics.png'))
    plt.close()

def recommend_cutoff_dates(cur):
    """Analyze data patterns to recommend appropriate cutoff dates"""
    print("\n" + "=" * 80)
    print("📅 CUTOFF DATE RECOMMENDATIONS")
    print("=" * 80)
    
    # Get key temporal milestones
    # Get all dates first
    cur.execute("""
        SELECT from_date
        FROM employees.department_employee
        WHERE from_date != '9999-01-01'
        ORDER BY from_date
    """)
    dates = [row[0] for row in cur.fetchall()]
    
    if dates:
        earliest_date = dates[0]
        latest_date = dates[-1]
        n = len(dates)
        q1_idx = n // 4
        median_idx = n // 2
        q3_idx = (3 * n) // 4
        
        q1_date = dates[q1_idx]
        median_date = dates[median_idx]
        q3_date = dates[q3_idx]
    else:
        earliest_date = None
        latest_date = None
        q1_date = None
        median_date = None
        q3_date = None
    
    # Get latest actual end date
    cur.execute("""
        SELECT MAX(to_date)
        FROM employees.department_employee
        WHERE to_date != '9999-01-01'
    """)
    latest_actual_date = cur.fetchone()[0]
    
    dates = (earliest_date, latest_actual_date, q1_date, median_date, q3_date)
    
    # Get event distribution
    cur.execute("""
        WITH events AS (
            -- Employment starts
            SELECT from_date as event_date, 'start' as event_type
            FROM employees.department_employee
            UNION ALL
            -- Employment ends (excluding current)
            SELECT to_date, 'end'
            FROM employees.department_employee
            WHERE to_date != '9999-01-01'
            UNION ALL
            -- Salary changes
            SELECT from_date, 'salary'
            FROM employees.salary
            UNION ALL
            -- Title changes
            SELECT from_date, 'title'
            FROM employees.title
        )
        SELECT 
            EXTRACT(YEAR FROM event_date) as year,
            event_type,
            COUNT(*) as count
        FROM events
        GROUP BY EXTRACT(YEAR FROM event_date), event_type
        ORDER BY year, event_type
    """)
    events = pd.DataFrame(cur.fetchall(), columns=['year', 'event_type', 'count'])
    
    print("\n📊 Data Timeline:")
    print(f"  • Earliest Date: {dates[0]}")
    print(f"  • Latest Actual Date: {dates[1]}")
    print(f"  • Q1 Date: {dates[2]}")
    print(f"  • Median Date: {dates[3]}")
    print(f"  • Q3 Date: {dates[4]}")
    
    # Calculate stable periods
    stable_start = dates[2]  # Q1 date
    stable_end = dates[4]    # Q3 date
    split1 = dates[2] + (dates[3] - dates[2])/2  # Between Q1 and median
    split2 = dates[3] + (dates[4] - dates[3])/2  # Between median and Q3
    
    print("\n📋 Recommended Cutoff Dates:")
    print(f"  • Training Cutoff  : {split1.strftime('%Y-%m-%d')}")
    print(f"    - Uses {events[events['year'] < split1.year]['count'].sum():,} events for training")
    print(f"  • Validation Cutoff: {dates[3].strftime('%Y-%m-%d')}")
    print(f"    - Uses {events[(events['year'] >= split1.year) & (events['year'] < dates[3].year)]['count'].sum():,} events for validation")
    print(f"  • Testing Cutoff   : {split2.strftime('%Y-%m-%d')}")
    print(f"    - Uses {events[(events['year'] >= dates[3].year) & (events['year'] < split2.year)]['count'].sum():,} events for testing")
    
    print("\n💡 Rationale:")
    print("  • Training period captures early patterns and baseline behavior")
    print("  • Validation period includes mix of stable and transition periods")
    print("  • Testing period represents mature data patterns")
    print("  • All periods have sufficient events for meaningful analysis")
    
    return split1, dates[3], split2

def check_requirements():
    """Check and install required packages"""
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'networkx': 'networkx',
        'psycopg2': 'psycopg2-binary'
    }
    print("📦 Checking required packages...")
    
    import subprocess
    import sys
    
    for module, package in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def save_analysis_summary(output_dir, content):
    """Save analysis summary to a text file"""
    summary_file = os.path.join(output_dir, "analysis_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(content)
    print(f"📝 Analysis summary saved to: {summary_file}")

def main():
    """Run comprehensive data analysis"""
    # Capture output for summary
    import sys
    from io import StringIO
    
    # Store original stdout
    original_stdout = sys.stdout
    output_capture = StringIO()
    sys.stdout = output_capture
    
    try:
        # Check requirements first
        check_requirements()
        # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Try several possible output locations
    possible_dirs = [
        os.path.join(os.getcwd(), "experiment_logs"),  # Current directory
        "/content/experiment_logs",                     # Colab root
        "/tmp/experiment_logs"                         # Fallback to /tmp
    ]
    
    # Find first writable directory
    output_base = None
    for d in possible_dirs:
        try:
            os.makedirs(d, exist_ok=True)
            # Test if writable
            test_file = os.path.join(d, "test.txt")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            output_base = d
            break
        except (OSError, IOError):
            continue
    
    if output_base is None:
        raise RuntimeError("Could not find a writable directory for output")
    
    # Create analysis-specific directory
    output_dir = os.path.join(output_base, f"data_analysis_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📁 Output will be saved to: {output_dir}")
    
    # Connect to database
    conn = connect_to_db()
    if not conn:
        return
    
    try:
        cur = conn.cursor()
        
        print("✅ Connected to PostgreSQL")
        print("🔍 Running comprehensive data analysis...\n")
        
        # Initialize visualizer
        viz = EDAVisualizer(output_dir)
        
        # Get table counts
        tables_info = {}
        for table in ['employee', 'department', 'department_employee', 
                     'department_manager', 'salary', 'title']:
            cur.execute(f"SELECT COUNT(*) FROM employees.{table}")
            count = cur.fetchone()[0]
            tables_info[table] = {'count': count}
            print(f"📊 {table}: {count:,} records")
        
        # Create entity relationship diagram
        viz.plot_entity_relationships(tables_info)
        
        # Run analyses
        analyze_temporal_distribution(cur, output_dir)
        analyze_department_patterns(cur, output_dir)
        analyze_salary_patterns(cur, output_dir)
        train_cutoff, val_cutoff, test_cutoff = recommend_cutoff_dates(cur)
        
        # Get data for correlation analysis
        cur.execute("""
            SELECT 
                e.id as employee_id,
                EXTRACT(YEAR FROM AGE(CURRENT_DATE, e.hire_date)) as tenure_years,
                s.amount as salary,
                t.duration_years,
                EXTRACT(YEAR FROM s.from_date) as year
            FROM employees.employee e
            JOIN employees.salary s ON e.id = s.employee_id
            JOIN (
                SELECT 
                    employee_id,
                    EXTRACT(YEAR FROM AGE(to_date, from_date)) as duration_years
                FROM employees.title
                WHERE to_date != '9999-01-01'
            ) t ON e.id = t.employee_id
            WHERE s.to_date = '9999-01-01'
        """)
        feature_data = pd.DataFrame(cur.fetchall(), 
                                  columns=['employee_id', 'tenure_years', 'salary', 
                                         'title_duration', 'year'])
        
        # Create correlation matrix
        viz.plot_correlation_matrix(feature_data)
        
        # Check for missing data
        viz.plot_missing_data(feature_data)
        
        # Create HTML report
        viz.create_html_report(f"Employee Data Analysis Report ({datetime.now().strftime('%Y-%m-%d')})")
        
        print("\n✨ Analysis complete!")
        print(f"📁 Results saved to: {output_dir}")
        print(f"📊 View the full report at: {os.path.join(output_dir, f'eda_report_{viz.timestamp}.html')}")
        
        # Save captured output
        sys.stdout = original_stdout
        output_content = output_capture.getvalue()
        save_analysis_summary(output_dir, output_content)
        
        # Print final messages to actual stdout
        print("\n✨ Analysis complete!")
        print(f"📁 Results saved to: {output_dir}")
        print(f"📊 View the full report at: {os.path.join(output_dir, f'eda_report_{viz.timestamp}.html')}")
        
        # Create summary file
        with open(os.path.join(output_dir, "recommended_settings.txt"), "w") as f:
            f.write("Recommended Settings for test_attrition.sh:\n\n")
            f.write("BASE_PARAMS=\"\n")
            f.write("    --hidden_dim 256 \\\n")
            f.write("    --num_layers 2 \\\n")
            f.write("    --dropout 0.5 \\\n")
            f.write("    --epochs 100 \\\n")
            f.write("    --batch_size 2048 \\\n")
            f.write("    --lr 0.001 \\\n")
            f.write("    --wandb True \\\n")
            f.write("    --wandb_project sql_to_gnn \\\n")
            f.write(f"    --train_cutoff {train_cutoff.strftime('%Y-%m-%d')} \\\n")
            f.write(f"    --val_cutoff {val_cutoff.strftime('%Y-%m-%d')} \\\n")
            f.write(f"    --test_cutoff {test_cutoff.strftime('%Y-%m-%d')}\"\n")
        
    finally:
        if conn:
            conn.close()
        sys.stdout = original_stdout
        output_capture.close()

if __name__ == "__main__":
    main()
