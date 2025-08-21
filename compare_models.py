import argparse
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_experiment_results(results_dir):
    """Load all experiment results from JSON files"""
    results = []
    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            with open(os.path.join(results_dir, filename), 'r') as f:
                data = json.load(f)
                # Extract experiment name and metrics
                exp_name = data['params'].get('experiment_name', 'unknown')
                metrics = {
                    'accuracy': float(data['metrics']['test']),
                    'auc': float(data['metrics'].get('test_auc', 0.0)),
                    'precision': float(data['metrics'].get('test_precision', 0.0)),
                    'recall': float(data['metrics'].get('test_recall', 0.0)),
                    'f1': float(data['metrics'].get('test_f1_macro', 0.0)),
                    'runtime': float(data['metrics']['runtime_seconds']),
                    'model_type': data['params']['model'],
                    'graph_type': data['params'].get('graph_type', 'bipartite'),
                    'timestamp': datetime.fromtimestamp(data['metrics']['training'][0]['timestamp'])
                }
                results.append({
                    'experiment': exp_name,
                    **metrics
                })
    return pd.DataFrame(results)

def plot_metrics_comparison(df, metrics, output_file):
    """Create comparison plots for different metrics"""
    # Set up the plot
    n_metrics = len(metrics)
    fig, axes = plt.subplots(2, (n_metrics + 1) // 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        sns.barplot(data=df, x='experiment', y=metric, ax=axes[i])
        axes[i].set_title(f'{metric.upper()} by Model Type')
        axes[i].tick_params(axis='x', rotation=45)
    
    # Remove empty subplots if any
    for i in range(n_metrics, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"📊 Comparison plots saved to {output_file}")
    
    # Print summary table
    print("\n📋 Model Performance Summary:")
    summary = df.groupby('graph_type')[metrics].mean()
    print(summary.round(4).to_string())

def main():
    parser = argparse.ArgumentParser(description='Compare GNN model performances')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing experiment JSON files')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Output file for comparison plots')
    parser.add_argument('--metrics', type=str, default='accuracy,auc,precision,recall,f1',
                       help='Comma-separated list of metrics to compare')
    
    args = parser.parse_args()
    metrics = args.metrics.split(',')
    
    # Load and process results
    results_df = load_experiment_results(args.results_dir)
    
    # Create comparison plots
    plot_metrics_comparison(results_df, metrics, args.output_file)
    
    # Additional analysis
    print("\n⚡ Key Findings:")
    best_model = results_df.loc[results_df['f1'].idxmax()]
    print(f"Best Overall Model: {best_model['experiment']} (F1: {best_model['f1']:.4f})")
    
    fastest_model = results_df.loc[results_df['runtime'].idxmin()]
    print(f"Fastest Model: {fastest_model['experiment']} (Runtime: {fastest_model['runtime']:.2f}s)")
    
    # Print recommendations
    print("\n💡 Recommendations:")
    if best_model['graph_type'] == 'heterogeneous':
        print("- The heterogeneous graph structure shows strong performance, suggesting rich relationship modeling is beneficial")
    elif best_model['graph_type'] == 'temporal':
        print("- Temporal patterns appear significant for attrition prediction")
    elif best_model['graph_type'] == 'hierarchical':
        print("- Organizational structure plays a key role in attrition")
    
    print("\n🎯 Next Steps:")
    print("1. Consider ensemble approach combining top performing models")
    print("2. Fine-tune hyperparameters of the best performing model")
    print("3. Analyze feature importance and relationship patterns")

if __name__ == "__main__":
    main()
