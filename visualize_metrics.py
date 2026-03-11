import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# 1. The Empirical Data (From Clinical Query 1)
tokens = [
    "yes", ",", "mit", "och", "ond", "ria", "play", "a", 
    "role", "in", "rem", "od", "eling", "la", "ce"
]
times = [69.54, 37.79, 30.83, 30.15, 30.12, 30.05, 29.99, 29.94, 
         12.92, 30.33, 30.29, 30.06, 12.95, 30.27, 29.95]
exit_types = [
    "Early (Forced)", "Full Pass", "Full Pass", "Full Pass", "Full Pass", 
    "Full Pass", "Full Pass", "Full Pass", "Early (Thresh)", "Full Pass", 
    "Full Pass", "Full Pass", "Early (Thresh)", "Full Pass", "Full Pass"
]

df = pd.DataFrame({
    'Token': tokens,
    'Time_ms': times,
    'Exit_Type': exit_types,
    'Token_Index': range(1, len(tokens) + 1)
})

# IEEE Paper Styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})

def plot_execution_timeline():
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Map colors to exit types for visual clarity
    colors = []
    for et in df['Exit_Type']:
        if "Full Pass" in et: colors.append('#2b5b84') # Dark Blue
        elif "Thresh" in et: colors.append('#2e8b57') # Sea Green
        else: colors.append('#d9534f') # Red (Forced/Prefill)
            
    bars = ax.bar(df['Token_Index'], df['Time_ms'], color=colors, edgecolor='black', linewidth=0.5)
    
    # Add the 45ms Hard Deadline Line
    ax.axhline(y=45.0, color='red', linestyle='--', linewidth=1.5, label='Hard Deadline (45.0 ms)')
    
    # Labeling
    ax.set_xticks(df['Token_Index'])
    ax.set_xticklabels(df['Token'], rotation=45, ha='right')
    ax.set_ylabel('Execution Time (ms)')
    ax.set_xlabel('Generated Tokens')
    ax.set_title('Dynamic Scheduler Token Execution Timeline')
    
    # Custom Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2b5b84', edgecolor='black', label='Full Pass (~30 ms)'),
        Patch(facecolor='#2e8b57', edgecolor='black', label='Early Exit (High Conf, ~13 ms)'),
        Patch(facecolor='#d9534f', edgecolor='black', label='Prefill Phase (TTFT)'),
        ax.lines[0] # The deadline line
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('execution_timeline.png', dpi=300, bbox_inches='tight')
    print("Saved 'execution_timeline.png'")

def plot_tail_latency_cdf():
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Remove the prefill token (Index 0) to evaluate just the generation phase (TPOT)
    tpot_times = np.sort(df['Time_ms'][1:])
    p = 1. * np.arange(len(tpot_times)) / (len(tpot_times) - 1)
    
    ax.plot(tpot_times, p, marker='o', linestyle='-', color='#2b5b84', linewidth=2)
    ax.axvline(x=45.0, color='red', linestyle='--', linewidth=1.5, label='Hard Deadline')
    
    ax.set_xlabel('Execution Time (ms)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('CDF of Token Generation Latency (TPOT)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('tail_latency_cdf.png', dpi=300, bbox_inches='tight')
    print("Saved 'tail_latency_cdf.png'")

if __name__ == "__main__":
    plot_execution_timeline()
    plot_tail_latency_cdf()