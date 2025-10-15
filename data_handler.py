import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_data(file_path="timeseries.csv"):
    """
    Load and clean telemetry data from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Cleaned and formatted DataFrame
        
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If the CSV format is invalid
    """
    try:
        # Load the CSV data
        df = pd.read_csv(file_path)
        
        # Validate required columns
        required_cols = ['Timestamp', 'VLAN', 'Input Rate (bps)', 'Input Packets', 
                        'Output Rate (bps)', 'Output Packets']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert timestamp to datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        
        # Remove rows with invalid timestamps
        df = df.dropna(subset=['Timestamp'])
        
        # Clean VLAN column - standardize format
        df['VLAN'] = df['VLAN'].astype(str).str.strip()
        df['VLAN'] = df['VLAN'].str.replace('Vlan', 'VLAN', case=False)
        
        # Ensure VLAN names are consistent
        df['VLAN'] = df['VLAN'].str.replace(r'VLAN0+', 'VLAN', regex=True)
        
        # Convert numeric columns to appropriate types
        numeric_cols = ['Input Rate (bps)', 'Input Packets', 'Output Rate (bps)', 'Output Packets']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with all NaN values in numeric columns
        df = df.dropna(subset=numeric_cols, how='all')
        
        # Sort by timestamp
        df = df.sort_values('Timestamp').reset_index(drop=True)
        
        # Add derived metrics
        df['Total Rate (bps)'] = df['Input Rate (bps)'] + df['Output Rate (bps)']
        df['Total Packets'] = df['Input Packets'] + df['Output Packets']
        df['Rate Ratio (Out/In)'] = df['Output Rate (bps)'] / df['Input Rate (bps)'].replace(0, np.nan)
        
        return df
        
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found at path: {file_path}")
    except Exception as e:
        raise ValueError(f"Error loading data: {str(e)}")


def get_data_summary(df):
    """
    Generate comprehensive summary statistics for the telemetry data.
    
    Args:
        df (pd.DataFrame): Telemetry data
        
    Returns:
        dict: Summary statistics
    """
    summary = {
        'total_records': len(df),
        'total_vlans': df['VLAN'].nunique(),
        'time_range': f"{(df['Timestamp'].max() - df['Timestamp'].min()).total_seconds() / 60:.1f} min",
        'avg_input_rate': df['Input Rate (bps)'].mean(),
        'avg_output_rate': df['Output Rate (bps)'].mean(),
        'max_input_rate': df['Input Rate (bps)'].max(),
        'max_output_rate': df['Output Rate (bps)'].max(),
        'vlan_stats': {}
    }
    
    # Per-VLAN statistics
    for vlan in df['VLAN'].unique():
        vlan_data = df[df['VLAN'] == vlan]
        summary['vlan_stats'][vlan] = {
            'avg_input_rate': float(vlan_data['Input Rate (bps)'].mean()),
            'avg_output_rate': float(vlan_data['Output Rate (bps)'].mean()),
            'max_input_rate': float(vlan_data['Input Rate (bps)'].max()),
            'max_output_rate': float(vlan_data['Output Rate (bps)'].max()),
            'total_packets_in': int(vlan_data['Input Packets'].sum()),
            'total_packets_out': int(vlan_data['Output Packets'].sum())
        }
    
    return summary


def get_anomalies(df, threshold=2.5):
    """
    Detect anomalies in the telemetry data using statistical methods.
    
    Args:
        df (pd.DataFrame): Telemetry data
        threshold (float): Number of standard deviations for anomaly detection
        
    Returns:
        pd.DataFrame: Rows containing anomalies
    """
    anomalies = []
    
    for vlan in df['VLAN'].unique():
        vlan_data = df[df['VLAN'] == vlan].copy()
        
        for metric in ['Input Rate (bps)', 'Output Rate (bps)']:
            mean = vlan_data[metric].mean()
            std = vlan_data[metric].std()
            
            if std > 0:  # Avoid division by zero
                vlan_data['z_score'] = np.abs((vlan_data[metric] - mean) / std)
                anomaly_rows = vlan_data[vlan_data['z_score'] > threshold].copy()
                anomaly_rows['anomaly_metric'] = metric
                anomaly_rows['anomaly_value'] = anomaly_rows[metric]
                anomalies.append(anomaly_rows)
    
    if anomalies:
        return pd.concat(anomalies)[['Timestamp', 'VLAN', 'anomaly_metric', 'anomaly_value', 'z_score']]
    else:
        return pd.DataFrame()


def plot_metric(df, vlan=None, metric="Input Rate (bps)", plot_type="line"):
    """
    Generate an enhanced plot for a specific metric.
    
    Args:
        df (pd.DataFrame): Telemetry data
        vlan (str, optional): Specific VLAN to plot. None for all VLANs
        metric (str): Metric to plot
        plot_type (str): Type of plot ('line', 'area', 'bar')
        
    Returns:
        io.BytesIO: Buffer containing the plot image
        
    Raises:
        ValueError: If metric is not in dataframe
    """
    # Validate metric
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in dataframe. Available: {list(df.columns)}")
    
    # Create figure with better aesthetics
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor('white')
    
    # Color palette
    colors = sns.color_palette("husl", n_colors=df['VLAN'].nunique())
    
    # Plot data based on VLAN selection
    if vlan:
        data = df[df['VLAN'] == vlan].copy()
        if data.empty:
            raise ValueError(f"No data found for VLAN: {vlan}")
        
        color = colors[0]
        
        if plot_type == "area":
            ax.fill_between(data['Timestamp'], data[metric], alpha=0.3, color=color)
            ax.plot(data['Timestamp'], data[metric], label=f"{vlan}", 
                   linewidth=2.5, marker='o', markersize=6, color=color)
        elif plot_type == "bar":
            ax.bar(data['Timestamp'], data[metric], label=f"{vlan}", 
                  color=color, alpha=0.7, width=0.002)
        else:  # line
            ax.plot(data['Timestamp'], data[metric], label=f"{vlan}", 
                   linewidth=2.5, marker='o', markersize=6, color=color)
        
        title = f"{metric} for {vlan}"
        
        # Add statistics annotation
        mean_val = data[metric].mean()
        max_val = data[metric].max()
        min_val = data[metric].min()
        
        stats_text = f"Mean: {mean_val:,.0f}\nMax: {max_val:,.0f}\nMin: {min_val:,.0f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', 
               facecolor='wheat', alpha=0.5), fontsize=10)
    else:
        # Plot all VLANs
        vlans = sorted(df['VLAN'].unique())
        for idx, v in enumerate(vlans):
            data = df[df['VLAN'] == v].copy()
            color = colors[idx % len(colors)]
            
            if plot_type == "area":
                ax.fill_between(data['Timestamp'], data[metric], alpha=0.2, color=color)
                ax.plot(data['Timestamp'], data[metric], label=v, 
                       linewidth=2, marker='o', markersize=4, color=color)
            elif plot_type == "bar":
                ax.bar(data['Timestamp'], data[metric], label=v, 
                      color=color, alpha=0.6, width=0.002)
            else:  # line
                ax.plot(data['Timestamp'], data[metric], label=v, 
                       linewidth=2, marker='o', markersize=4, color=color)
        
        title = f"{metric} - All VLANs Comparison"
    
    # Enhanced formatting
    ax.set_xlabel("Time", fontsize=12, fontweight='bold')
    ax.set_ylabel(metric, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Improved legend
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), 
             frameon=True, shadow=True, fontsize=10)
    
    # Grid styling
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
    ax.set_axisbelow(True)
    
    # Format y-axis with comma separators
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    # Rotate and align x-axis labels
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add horizontal line for mean
    if vlan:
        ax.axhline(y=mean_val, color='red', linestyle='--', 
                  linewidth=1, alpha=0.5, label=f'Mean: {mean_val:,.0f}')
    
    # Tight layout
    plt.tight_layout()
    
    # Save to buffer with high quality
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=150, 
               facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    
    return buf


def plot_comparison(df, metric="Input Rate (bps)"):
    """
    Create a comparison plot showing all VLANs side by side.
    
    Args:
        df (pd.DataFrame): Telemetry data
        metric (str): Metric to compare
        
    Returns:
        io.BytesIO: Buffer containing the plot image
    """
    vlans = sorted(df['VLAN'].unique())
    n_vlans = len(vlans)
    
    fig, axes = plt.subplots(n_vlans, 1, figsize=(14, 5*n_vlans), sharex=True)
    if n_vlans == 1:
        axes = [axes]
    
    colors = sns.color_palette("husl", n_colors=n_vlans)
    
    for idx, (vlan, ax) in enumerate(zip(vlans, axes)):
        data = df[df['VLAN'] == vlan]
        color = colors[idx]
        
        ax.plot(data['Timestamp'], data[metric], linewidth=2, 
               marker='o', markersize=5, color=color, label=vlan)
        ax.fill_between(data['Timestamp'], data[metric], alpha=0.3, color=color)
        
        ax.set_ylabel(metric, fontsize=11, fontweight='bold')
        ax.set_title(f"{vlan} - {metric}", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right')
        
        # Format y-axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    axes[-1].set_xlabel("Time", fontsize=12, fontweight='bold')
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=150)
    buf.seek(0)
    plt.close(fig)
    
    return buf


def export_summary_report(df, output_file="telemetry_report.txt"):
    """
    Export a comprehensive text report of telemetry analysis.
    
    Args:
        df (pd.DataFrame): Telemetry data
        output_file (str): Output file path
    """
    with open(output_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("TELEMETRY DATA ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        summary = get_data_summary(df)
        
        f.write("OVERVIEW\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total Records: {summary['total_records']}\n")
        f.write(f"VLANs Monitored: {summary['total_vlans']}\n")
        f.write(f"Time Range: {summary['time_range']}\n")
        f.write(f"Average Input Rate: {summary['avg_input_rate']:,.0f} bps\n")
        f.write(f"Average Output Rate: {summary['avg_output_rate']:,.0f} bps\n")
        f.write(f"Peak Input Rate: {summary['max_input_rate']:,.0f} bps\n")
        f.write(f"Peak Output Rate: {summary['max_output_rate']:,.0f} bps\n\n")
        
        f.write("PER-VLAN STATISTICS\n")
        f.write("-" * 70 + "\n")
        for vlan, stats in summary['vlan_stats'].items():
            f.write(f"\n{vlan}:\n")
            f.write(f"  Avg Input Rate: {stats['avg_input_rate']:,.0f} bps\n")
            f.write(f"  Avg Output Rate: {stats['avg_output_rate']:,.0f} bps\n")
            f.write(f"  Max Input Rate: {stats['max_input_rate']:,.0f} bps\n")
            f.write(f"  Max Output Rate: {stats['max_output_rate']:,.0f} bps\n")
            f.write(f"  Total Packets In: {stats['total_packets_in']:,}\n")
            f.write(f"  Total Packets Out: {stats['total_packets_out']:,}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")
    
    print(f"Report exported to: {output_file}")