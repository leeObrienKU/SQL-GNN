#!/bin/bash

# Default values
CUTOFF_DATE="2002-12-31"
MAX_EMPLOYEES=500
OUTPUT_FILE="employee_department_graph.png"

# Help message
show_help() {
    echo "Usage: ./visualize_graph.sh [options]"
    echo "Options:"
    echo "  -c, --cutoff DATE    Cutoff date (default: 2002-12-31)"
    echo "  -n, --num-emp NUM    Number of employees to sample (default: 500)"
    echo "  -o, --output FILE    Output file name (default: employee_department_graph.png)"
    echo "  -h, --help          Show this help message"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--cutoff)
            CUTOFF_DATE="$2"
            shift 2
            ;;
        -n|--num-emp)
            MAX_EMPLOYEES="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Run visualization
echo "🎨 Creating graph visualization..."
echo "• Cutoff date: $CUTOFF_DATE"
echo "• Sample size: $MAX_EMPLOYEES employees"
echo "• Output file: $OUTPUT_FILE"

python - << EOF
from simple_graph_viz import visualize_graph_simple
plt = visualize_graph_simple(
    cutoff_date="$CUTOFF_DATE",
    max_employees=$MAX_EMPLOYEES
)
plt.savefig('$OUTPUT_FILE', dpi=300, bbox_inches='tight')
EOF

echo "✅ Graph saved as $OUTPUT_FILE"
