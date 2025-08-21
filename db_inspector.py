import psycopg2
import pandas as pd
import numpy as np
from tabulate import tabulate
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def connect_to_db():
    """Establish database connection"""
    try:
        conn = psycopg2.connect(
            dbname="empdb",
            user="postgres",
            host="/var/run/postgresql"
        )
        return conn
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return None

def get_table_overview(cur):
    """Get overview of all tables in employees schema"""
    print("=" * 80)
    print("📊 DATABASE OVERVIEW")
    print("=" * 80)
    
    cur.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'employees'
        ORDER BY table_name
    """)
    tables = [x[0] for x in cur.fetchall()]
    
    print("Tables in 'employees' schema:")
    for i, table in enumerate(tables, 1):
        print(f"  {i}. {table}")
    print()
    return tables

def analyze_employee_demographics(cur):
    """Analyze employee demographics and temporal patterns"""
    print("=" * 80)
    print("👥 EMPLOYEE DEMOGRAPHICS & TEMPORAL ANALYSIS")
    print("=" * 80)
    
    # Total employee count
    cur.execute("SELECT COUNT(*) FROM employees.employee")
    total_employees = cur.fetchone()[0]
    print(f"Total Employees: {total_employees:,}")
    
    # Gender distribution
    cur.execute("""
        SELECT gender, COUNT(*) as count, 
               ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
        FROM employees.employee 
        GROUP BY gender
        ORDER BY count DESC
    """)
    gender_stats = cur.fetchall()
    print("\nGender Distribution:")
    print(tabulate(gender_stats, headers=["Gender", "Count", "Percentage (%)"], tablefmt="grid"))
    
    # Birth date analysis
    cur.execute("""
        SELECT 
            MIN(birth_date) as earliest_birth,
            MAX(birth_date) as latest_birth,
            AVG(EXTRACT(YEAR FROM CURRENT_DATE) - EXTRACT(YEAR FROM birth_date)) as avg_age,
            STDDEV(EXTRACT(YEAR FROM CURRENT_DATE) - EXTRACT(YEAR FROM birth_date)) as age_std
        FROM employees.employee
    """)
    birth_stats = cur.fetchone()
    print(f"\nBirth Date Range: {birth_stats[0]} to {birth_stats[1]}")
    print(f"Average Age: {birth_stats[2]:.1f} years (±{birth_stats[3]:.1f})")
    
    # Hire date analysis
    cur.execute("""
        SELECT 
            MIN(hire_date) as earliest_hire,
            MAX(hire_date) as latest_hire,
            AVG(EXTRACT(YEAR FROM CURRENT_DATE) - EXTRACT(YEAR FROM hire_date)) as avg_tenure,
            STDDEV(EXTRACT(YEAR FROM CURRENT_DATE) - EXTRACT(YEAR FROM hire_date)) as tenure_std
        FROM employees.employee
    """)
    hire_stats = cur.fetchone()
    print(f"\nHire Date Range: {hire_stats[0]} to {hire_stats[1]}")
    print(f"Average Tenure: {hire_stats[2]:.1f} years (±{hire_stats[3]:.1f})")
    
    # Temporal distribution by year
    cur.execute("""
        SELECT 
            EXTRACT(YEAR FROM hire_date) as hire_year,
            COUNT(*) as new_hires
        FROM employees.employee
        GROUP BY EXTRACT(YEAR FROM hire_date)
        ORDER BY hire_year
    """)
    yearly_hires = cur.fetchall()
    print("\nYearly Hiring Pattern:")
    print(tabulate(yearly_hires, headers=["Year", "New Hires"], tablefmt="grid"))

def analyze_departments(cur):
    """Analyze department structure and employee distribution"""
    print("\n" + "=" * 80)
    print("🏢 DEPARTMENT ANALYSIS")
    print("=" * 80)
    
    # Department employee counts
    cur.execute("""
        SELECT 
            d.dept_name,
            COUNT(de.employee_id) as employee_count,
            ROUND(COUNT(de.employee_id) * 100.0 / SUM(COUNT(de.employee_id)) OVER (), 2) as percentage
        FROM employees.department d
        LEFT JOIN employees.department_employee de ON d.id = de.department_id
        WHERE de.to_date IS NULL OR de.to_date > CURRENT_DATE
        GROUP BY d.id, d.dept_name
        ORDER BY employee_count DESC
    """)
    dept_stats = cur.fetchall()
    print("Current Department Sizes:")
    print(tabulate(dept_stats, headers=["Department", "Employees", "Percentage (%)"], tablefmt="grid"))

def analyze_salary_distribution(cur):
    """Analyze salary patterns and distributions"""
    print("\n" + "=" * 80)
    print("💰 SALARY ANALYSIS")
    print("=" * 80)
    
    # Current salary statistics
    cur.execute("""
        SELECT 
            COUNT(*) as salary_records,
            MIN(amount) as min_salary,
            MAX(amount) as max_salary,
            AVG(amount) as avg_salary,
            STDDEV(amount) as salary_std,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY amount) as q1,
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY amount) as median,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY amount) as q3
        FROM employees.salary
        WHERE to_date IS NULL OR to_date > CURRENT_DATE
    """)
    salary_stats = cur.fetchone()
    
    print("Current Salary Distribution:")
    print(f"  Records: {salary_stats[0]:,}")
    print(f"  Range: ${salary_stats[1]:,.0f} - ${salary_stats[2]:,.0f}")
    print(f"  Mean: ${salary_stats[3]:,.0f} (±${salary_stats[4]:,.0f})")
    print(f"  Q1: ${salary_stats[5]:,.0f}")
    print(f"  Median: ${salary_stats[6]:,.0f}")
    print(f"  Q3: ${salary_stats[7]:,.0f}")
    
    # Salary by department
    cur.execute("""
        SELECT 
            d.dept_name,
            COUNT(s.employee_id) as employees,
            ROUND(AVG(s.amount), 2) as avg_salary,
            ROUND(STDDEV(s.amount), 2) as salary_std,
            ROUND(MIN(s.amount), 2) as min_salary,
            ROUND(MAX(s.amount), 2) as max_salary
        FROM employees.department d
        JOIN employees.department_employee de ON d.id = de.department_id
        JOIN employees.salary s ON de.employee_id = s.employee_id
        WHERE (de.to_date IS NULL OR de.to_date > CURRENT_DATE)
          AND (s.to_date IS NULL OR s.to_date > CURRENT_DATE)
        GROUP BY d.id, d.dept_name
        ORDER BY avg_salary DESC
    """)
    dept_salary = cur.fetchall()
    print("\nSalary by Department:")
    print(tabulate(dept_salary, headers=["Department", "Employees", "Avg Salary", "Std Dev", "Min", "Max"], tablefmt="grid"))

def analyze_attrition_patterns(cur, cutoff_date="2000-01-01"):
    """Analyze attrition patterns and class imbalance"""
    print("\n" + "=" * 80)
    print("🚪 ATTRITION ANALYSIS")
    print("=" * 80)
    
    # First, let's see the actual date ranges in the data
    print("📅 DATA TEMPORAL RANGE ANALYSIS:")
    print("-" * 50)
    
    # Department employment date ranges
    cur.execute("""
        SELECT 
            MIN(from_date) as earliest_date,
            MAX(to_date) as latest_date,
            COUNT(*) as total_records,
            COUNT(CASE WHEN to_date IS NOT NULL THEN 1 END) as records_with_end_date
        FROM employees.department_employee
    """)
    dept_date_range = cur.fetchone()
    print(f"Department Employment Records:")
    print(f"  • Date Range: {dept_date_range[0]} to {dept_date_range[1]}")
    print(f"  • Total Records: {dept_date_range[2]:,}")
    print(f"  • Records with End Date: {dept_date_range[3]:,}")
    
    # Salary date ranges
    cur.execute("""
        SELECT 
            MIN(from_date) as earliest_salary,
            MAX(to_date) as latest_salary,
            COUNT(*) as total_salary_records
        FROM employees.salary
    """)
    salary_date_range = cur.fetchone()
    print(f"Salary Records:")
    print(f"  • Date Range: {salary_date_range[0]} to {salary_date_range[1]}")
    print(f"  • Total Records: {salary_date_range[2]:,}")
    
    # Title date ranges
    cur.execute("""
        SELECT 
            MIN(from_date) as earliest_title,
            MAX(to_date) as latest_title,
            COUNT(*) as total_title_records
        FROM employees.title
    """)
    title_date_range = cur.fetchone()
    print(f"Title Records:")
    print(f"  • Date Range: {title_date_range[0]} to {title_date_range[1]}")
    print(f"  • Total Records: {title_date_range[2]:,}")
    
    print(f"\nCurrent Analysis Cutoff: {cutoff_date}")
    print(f"Data Coverage: {dept_date_range[0]} to {dept_date_range[1]} ({dept_date_range[1] - dept_date_range[0] if dept_date_range[1] and dept_date_range[0] else 'Unknown'} days)")
    print("-" * 50)
    
    # Attrition definition: employees whose latest department assignment ended before cutoff
    cur.execute("""
        WITH latest_dept AS (
            SELECT 
                employee_id,
                department_id,
                to_date,
                ROW_NUMBER() OVER (PARTITION BY employee_id ORDER BY to_date DESC) as rn
            FROM employees.department_employee
        ),
        attrition_status AS (
            SELECT 
                e.id as employee_id,
                e.gender,
                e.hire_date,
                ld.to_date as last_dept_date,
                CASE 
                    WHEN ld.to_date < %s THEN 1 
                    ELSE 0 
                END as is_leaver
            FROM employees.employee e
            LEFT JOIN latest_dept ld ON e.id = ld.employee_id AND ld.rn = 1
        )
        SELECT 
            is_leaver,
            COUNT(*) as count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
        FROM attrition_status
        GROUP BY is_leaver
        ORDER BY is_leaver
    """, (cutoff_date,))
    
    attrition_stats = cur.fetchall()
    print(f"Attrition Analysis (Cutoff: {cutoff_date}):")
    print(tabulate(attrition_stats, headers=["Status", "Count", "Percentage (%)"], tablefmt="grid"))
    
    # Calculate class imbalance metrics
    stayers = attrition_stats[0][1] if attrition_stats[0][0] == 0 else attrition_stats[1][1]
    leavers = attrition_stats[1][1] if attrition_stats[1][0] == 1 else attrition_stats[0][1]
    
    imbalance_ratio = stayers / leavers
    print(f"\nClass Imbalance Metrics:")
    print(f"  Stayers:Leavers Ratio: {imbalance_ratio:.2f}:1")
    print(f"  Minority Class (Leavers): {leavers:,} ({leavers/(stayers+leavers)*100:.1f}%)")
    print(f"  Majority Class (Stayers): {stayers:,} ({stayers/(stayers+leavers)*100:.1f}%)")
    
    # Attrition by department
    cur.execute("""
        WITH latest_dept AS (
            SELECT 
                employee_id,
                department_id,
                to_date,
                ROW_NUMBER() OVER (PARTITION BY employee_id ORDER BY to_date DESC) as rn
            FROM employees.department_employee
        ),
        attrition_by_dept AS (
            SELECT 
                d.dept_name,
                COUNT(*) as total_employees,
                SUM(CASE WHEN ld.to_date < %s THEN 1 ELSE 0 END) as leavers,
                SUM(CASE WHEN ld.to_date >= %s OR ld.to_date IS NULL THEN 1 ELSE 0 END) as stayers
            FROM employees.department d
            JOIN latest_dept ld ON d.id = ld.department_id AND ld.rn = 1
            GROUP BY d.id, d.dept_name
        )
        SELECT 
            dept_name,
            total_employees,
            leavers,
            stayers,
            ROUND(leavers * 100.0 / total_employees, 2) as attrition_rate
        FROM attrition_by_dept
        ORDER BY attrition_rate DESC
    """, (cutoff_date, cutoff_date))
    
    dept_attrition = cur.fetchall()
    print(f"\nAttrition by Department:")
    print(tabulate(dept_attrition, headers=["Department", "Total", "Leavers", "Stayers", "Attrition Rate (%)"], tablefmt="grid"))
    
    # Attrition by tenure
    cur.execute("""
        WITH latest_dept AS (
            SELECT 
                employee_id,
                department_id,
                to_date,
                ROW_NUMBER() OVER (PARTITION BY employee_id ORDER BY to_date DESC) as rn
            FROM employees.department_employee
        ),
        tenure_attrition AS (
            SELECT 
                e.id as employee_id,
                e.hire_date,
                ld.to_date as last_dept_date,
                EXTRACT(YEAR FROM AGE(%s::date, e.hire_date)) as tenure_years,
                CASE 
                    WHEN ld.to_date < %s THEN 1 
                    ELSE 0 
                END as is_leaver
            FROM employees.employee e
            LEFT JOIN latest_dept ld ON e.id = ld.employee_id AND ld.rn = 1
        ),
        tenure_buckets AS (
            SELECT 
                CASE 
                    WHEN tenure_years < 2 THEN '0-2 years'
                    WHEN tenure_years < 5 THEN '2-5 years'
                    WHEN tenure_years < 10 THEN '5-10 years'
                    WHEN tenure_years < 20 THEN '10-20 years'
                    ELSE '20+ years'
                END as tenure_bucket,
                COUNT(*) as total_employees,
                SUM(is_leaver) as leavers
            FROM tenure_attrition
            GROUP BY 
                CASE 
                    WHEN tenure_years < 2 THEN '0-2 years'
                    WHEN tenure_years < 5 THEN '2-5 years'
                    WHEN tenure_years < 10 THEN '5-10 years'
                    WHEN tenure_years < 20 THEN '10-20 years'
                    ELSE '20+ years'
                END
        )
        SELECT 
            tenure_bucket,
            total_employees,
            leavers,
            ROUND(leavers * 100.0 / total_employees, 2) as attrition_rate
        FROM tenure_buckets
        ORDER BY 
            CASE tenure_bucket
                WHEN '0-2 years' THEN 1
                WHEN '2-5 years' THEN 2
                WHEN '5-10 years' THEN 3
                WHEN '10-20 years' THEN 4
                ELSE 5
            END
    """, (cutoff_date, cutoff_date))
    
    tenure_attrition = cur.fetchall()
    print(f"\nAttrition by Tenure:")
    print(tabulate(tenure_attrition, headers=["Tenure", "Total", "Leavers", "Attrition Rate (%)"], tablefmt="grid"))

def analyze_temporal_patterns(cur):
    """Analyze temporal patterns in the data"""
    print("\n" + "=" * 80)
    print("⏰ TEMPORAL PATTERN ANALYSIS")
    print("=" * 80)
    
    # Hiring patterns over time
    cur.execute("""
        SELECT 
            EXTRACT(YEAR FROM hire_date) as year,
            COUNT(*) as new_hires,
            COUNT(CASE WHEN gender = 'M' THEN 1 END) as male_hires,
            COUNT(CASE WHEN gender = 'F' THEN 1 END) as female_hires
        FROM employees.employee
        GROUP BY EXTRACT(YEAR FROM hire_date)
        ORDER BY year
    """)
    yearly_hires = cur.fetchall()
    print("Hiring Patterns by Year:")
    print(tabulate(yearly_hires, headers=["Year", "Total Hires", "Male", "Female"], tablefmt="grid"))
    
    # Salary changes over time
    cur.execute("""
        SELECT 
            EXTRACT(YEAR FROM from_date) as year,
            COUNT(*) as salary_changes,
            ROUND(AVG(amount), 2) as avg_salary,
            ROUND(STDDEV(amount), 2) as salary_std,
            ROUND(MIN(amount), 2) as min_salary,
            ROUND(MAX(amount), 2) as max_salary
        FROM employees.salary
        GROUP BY EXTRACT(YEAR FROM from_date)
        ORDER BY year
    """)
    yearly_salary = cur.fetchall()
    print("\nSalary Changes by Year:")
    print(tabulate(yearly_salary, headers=["Year", "Changes", "Avg Salary", "Std Dev", "Min", "Max"], tablefmt="grid"))
    
    # Department changes over time
    cur.execute("""
        SELECT 
            EXTRACT(YEAR FROM from_date) as year,
            COUNT(*) as dept_changes,
            COUNT(DISTINCT employee_id) as employees_affected
        FROM employees.department_employee
        GROUP BY EXTRACT(YEAR FROM from_date)
        ORDER BY year
    """)
    yearly_dept_changes = cur.fetchall()
    print("\nDepartment Changes by Year:")
    print(tabulate(yearly_dept_changes, headers=["Year", "Department Changes", "Employees Affected"], tablefmt="grid"))
    
    # Title changes over time
    cur.execute("""
        SELECT 
            EXTRACT(YEAR FROM from_date) as year,
            COUNT(*) as title_changes,
            COUNT(DISTINCT employee_id) as employees_promoted
        FROM employees.title
        GROUP BY EXTRACT(YEAR FROM from_date)
        ORDER BY year
    """)
    yearly_titles = cur.fetchall()
    print("\nTitle Changes by Year:")
    print(tabulate(yearly_titles, headers=["Year", "Title Changes", "Employees Promoted"], tablefmt="grid"))

def generate_summary_report(cur):
    """Generate a comprehensive summary report"""
    print("\n" + "=" * 80)
    print("📋 EXECUTIVE SUMMARY")
    print("=" * 80)
    
    # Get key metrics
    cur.execute("SELECT COUNT(*) FROM employees.employee")
    total_emp = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM employees.department")
    total_dept = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM employees.salary")
    total_salary_records = cur.fetchone()[0]
    
    cur.execute("""
        SELECT COUNT(*) FROM employees.department_employee 
        WHERE to_date < '2000-01-01'
    """)
    total_leavers = cur.fetchone()[0]
    
    print(f"Dataset Overview:")
    print(f"  • Total Employees: {total_emp:,}")
    print(f"  • Total Departments: {total_dept}")
    print(f"  • Salary Records: {total_salary_records:,}")
    print(f"  • Historical Leavers (pre-2000): {total_leavers:,}")
    print(f"  • Data Coverage: Comprehensive employee lifecycle data")
    print(f"  • Temporal Range: Historical data suitable for attrition prediction")
    print(f"  • Class Imbalance: Significant (stayers vs leavers)")
    print(f"  • Key Features: Demographics, salary, department, temporal patterns")

def save_eda_report(cur, filename="eda_report.txt"):
    """Save EDA report to file"""
    import sys
    from io import StringIO
    
    # Capture all output
    old_stdout = sys.stdout
    captured_output = StringIO()
    sys.stdout = captured_output
    
    try:
        # Run all analysis functions
        get_table_overview(cur)
        analyze_employee_demographics(cur)
        analyze_departments(cur)
        analyze_salary_distribution(cur)
        analyze_attrition_patterns(cur)
        analyze_temporal_patterns(cur)
        generate_summary_report(cur)
        
        # Get captured output
        report_content = captured_output.getvalue()
        
        # Save to file
        with open(filename, 'w') as f:
            f.write(report_content)
        
        print(f"\n✅ EDA report saved to: {filename}")
        
    finally:
        sys.stdout = old_stdout
        captured_output.close()

def inspect_database(cutoff_date="2000-01-01"):
    """Main function to run comprehensive EDA"""
    conn = connect_to_db()
    if not conn:
        return
    
    try:
        cur = conn.cursor()
        
        print("✅ Connected to PostgreSQL")
        print("🔍 Running comprehensive EDA for attrition analysis...\n")
        print(f"📅 Analysis Configuration:")
        print(f"   • Cutoff Date: {cutoff_date}")
        print(f"   • This date separates 'stayers' (still employed) from 'leavers' (left before this date)")
        print()
        
        # Run all analyses
        get_table_overview(cur)
        analyze_employee_demographics(cur)
        analyze_departments(cur)
        analyze_salary_distribution(cur)
        analyze_attrition_patterns(cur, cutoff_date)
        analyze_temporal_patterns(cur)
        generate_summary_report(cur)
        
        # Save report
        save_eda_report(cur)
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()

if __name__ == "__main__":
    import sys
    
    # Allow command line argument for cutoff date
    cutoff = "2000-01-01"  # default
    if len(sys.argv) > 1:
        cutoff = sys.argv[1]
        print(f"Using cutoff date: {cutoff}")
    
    inspect_database(cutoff)

