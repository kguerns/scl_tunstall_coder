# After running eval.py at different codelengths
# to generate results_codelength_*.txt result files, 
# Run 'python visualize.py' to create plots

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import re
import sys
import glob
import io 

# Data Parsing Functions

def parse_file_content(file_content: str, code_length: int) -> list:
    """
    Parses the content of a single results file for all file types.
    Returns a list of dictionaries (records).
    """
    records = []
    
    # Split the file content by the separator for each file block
    file_blocks = file_content.split('--- File:')
    
    # Assume 8 bits per original symbol for compression ratio calculation
    BITS_PER_ORIGINAL_SYMBOL = 8
    
    for file_block in file_blocks:
        if not file_block.strip() or 'Entropy:' not in file_block:
            continue

        try:
            # 1. Extract File Name
            file_name_match = re.match(r' (.*) ---\n', file_block)
            if not file_name_match:
                continue
            file_name = file_name_match.group(1).strip()
            
            # Use regex to extract numeric metrics
            entropy = float(re.search(r'Entropy:\s*([\d\.]+)', file_block).group(1))
            avg_bits = float(re.search(r'Avg Bits/Sym:\s*([\d\.]+)', file_block).group(1))
            encoding_time = float(re.search(r'Encoding Time \(ms\):\s*([\d\.]+)', file_block).group(1))
            
            # Extract decoding times (handle Serial and Parallel specifically)
            serial_time_match = re.search(r'Serial Decoder\s*:\s*([\d\.]+)', file_block)
            parallel_time_match = re.search(r'Parallel Decoder\s*:\s*([\d\.]+)', file_block)

            if not serial_time_match or not parallel_time_match:
                 raise ValueError("Missing Serial or Parallel Decoder time.")
                 
            serial_time = float(serial_time_match.group(1))
            parallel_time = float(parallel_time_match.group(1))
            
            # Calculate speedup
            speedup_factor = serial_time / parallel_time if parallel_time > 0 else 1.0
            
            # Calculate Compression Ratio (CR)
            compression_ratio = BITS_PER_ORIGINAL_SYMBOL / avg_bits

            # Append record
            records.append({
                'File': file_name,
                'CodeLength': code_length,
                'Entropy': entropy,
                'AvgBits': avg_bits,
                'CompressionRatio': compression_ratio,
                'EncodingTime_ms': encoding_time,
                'SerialTime_ms': serial_time,
                'ParallelTime_ms': parallel_time,
                'Speedup': speedup_factor,
            })

        except (AttributeError, ValueError) as e:
            # Catches if a specific regex search failed or conversion failed
            print(f"Warning: Failed to parse data for file {file_name} in CL={code_length}. Skipping. Error: {e}", file=sys.stderr)
            continue
            
    return records

def read_and_process_all_results(directory='.') -> pd.DataFrame:
    """
    Reads all result files in the given directory, extracts the code length
    from the filename, and combines all data into a single DataFrame.
    """
    all_records = []
    
    result_files = sorted(glob.glob(os.path.join(directory, 'results_codelength_*.txt')))
    
    if not result_files:
        print("Error: No result files found matching 'results_codelength_*.txt' in the current directory.", file=sys.stderr)
        return pd.DataFrame()

    for file_path in result_files:
        cl_match = re.search(r'codelength_(\d+)\.txt', file_path)
        if not cl_match:
            print(f"Warning: Could not determine code length for {file_path}. Skipping.", file=sys.stderr)
            continue
            
        code_length = int(cl_match.group(1))
        
        try:
            with open(file_path, 'r') as f:
                file_content = f.read()
            
            records = parse_file_content(file_content, code_length)
            all_records.extend(records)
            
        except Exception as e:
            print(f"Error reading or parsing file {file_path}: {e}", file=sys.stderr)
            continue

    return pd.DataFrame(all_records)


# Visualization Function

def generate_plots(df: pd.DataFrame):
    """
    Generates four key visualizations from the combined DataFrame.
    """
    if df.empty:
        print("No data to plot.", file=sys.stderr)
        return

    sns.set_theme(style="whitegrid")
    
    # --- Plot 1: Code Length Optimization (Avg Bits vs. CL) ---
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    
    # Plot AvgBits for each file as a line, showing the convergence.
    sns.lineplot(
        x='CodeLength', 
        y='AvgBits', 
        hue='File', 
        marker='o', 
        data=df, 
        palette='tab10', 
        ax=ax1
    )

    # Plot Entropy as a horizontal dashed line for each file (only draw once)
    entropy_map = df[['File', 'Entropy']].drop_duplicates().set_index('File')['Entropy']
    for file_name, entropy in entropy_map.items():
        # Find the correct line color from the legend mapping
        try:
            line = ax1.get_lines()[df['File'].unique().tolist().index(file_name)]
            color = line.get_color()
        except IndexError:
            # Fallback if line wasn't drawn (shouldn't happen if data is present)
            color = 'gray' 

        ax1.axhline(entropy, color=color, linestyle='--', alpha=0.6, linewidth=1)

    ax1.set_title('Compression Efficiency: Avg Bits vs. Code Length (L)', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Code Length L', fontsize=12)
    ax1.set_ylabel('Avg Bits/Symbol (Lower is Better)', fontsize=12)
    ax1.set_xticks(df['CodeLength'].unique())
    
    # Guidance on Optimal CL: "Should I test for more codelengths for certain files?"
    
    # Identify files that are furthest from their respective entropy limit (high entropy files)
    df['EfficiencyDelta'] = df['AvgBits'] - df['Entropy']
    df_max_delta = df.loc[df.groupby('File')['CodeLength'].idxmax()]
    
    high_entropy_files = df_max_delta.sort_values(by='EfficiencyDelta', ascending=False).head(2)['File'].tolist()
    
    suggestion = f"Suggestion: Test higher L for '{' & '.join(high_entropy_files)}'"
    
    # ax1.text(
    #     df['CodeLength'].max() + 0.1, df['AvgBits'].min(), 
    #     #f"Optimal L is where line approaches dashed Entropy line.\n{suggestion}", 
    #     f"Optimal L is where line approaches dashed Entropy line.",
    #     fontsize=10, color='gray', bbox=dict(boxstyle="round,pad=0.5", fc="aliceblue", alpha=0.8)
    # )
    
    print("Saving Code Length Optimization plots to tunstall_codelength_optimization.png")
    fig1.savefig('tunstall_codelength_optimization.png', bbox_inches='tight', dpi=300)
    plt.close(fig1)

    # --- Plot 2: Compression Ratio (CR) vs. Code Length (L) ---
    fig2, ax2 = plt.subplots(figsize=(12, 7))

    sns.lineplot(
        x='CodeLength', 
        y='CompressionRatio', 
        hue='File', 
        marker='o', 
        data=df, 
        palette='tab10', 
        ax=ax2
    )

    ax2.set_title('Compression Ratio (CR) vs. Code Length (L)', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Code Length L', fontsize=12)
    ax2.set_ylabel('Compression Ratio (Original 8 Bits / Avg Bits)', fontsize=12)
    ax2.set_xticks(df['CodeLength'].unique())
    
    # Add labels showing the best compression ratio achieved for each file
    df_max_cr = df.loc[df.groupby('File')['CompressionRatio'].idxmax()]
    for _, row in df_max_cr.iterrows():
        ax2.annotate(
            f'{row.CompressionRatio:.2f}:1',
            (row.CodeLength, row.CompressionRatio),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=9,
            fontweight='bold'
        )

    print("Saving Compression Ratio plots to tunstall_compression_ratio.png")
    fig2.savefig('tunstall_compression_ratio.png', bbox_inches='tight', dpi=300)
    plt.close(fig2)

    # --- Plot 3: Decoding Speedup by Code Length (Parallel vs. Serial) ---
    fig3, ax3 = plt.subplots(figsize=(12, 7))
    
    # Plot individual speedup curves (noisy lines) with transparency
    sns.lineplot(
        x='CodeLength', 
        y='Speedup', 
        hue='File', 
        marker='o', 
        data=df, 
        palette='tab10', 
        alpha=0.4,
        ax=ax3
    )

    # Plot the MEAN speedup across all files for clarity (The actual smooth trend)
    sns.lineplot(
        x='CodeLength', 
        y='Speedup', 
        data=df.groupby('CodeLength', as_index=False)['Speedup'].mean(),
        marker='X', 
        color='black', 
        linestyle='--', 
        linewidth=2, 
        label='Mean Speedup Across All Files', 
        ax=ax3
    )
    
    ax3.set_title('Decoding Speedup Factor vs. Code Length', fontsize=16, fontweight='bold')
    ax3.set_xlabel('Code Length L', fontsize=12)
    ax3.set_ylabel('Speedup Factor (x) [Serial/Parallel]', fontsize=12)
    ax3.set_xticks(df['CodeLength'].unique())
    ax3.axhline(df['Speedup'].mean(), color='red', linestyle=':', label='Overall Average Speedup (Global)')
    
    # Adjust legend to show only the File names and the Mean Speedup line clearly
    handles, labels = ax3.get_legend_handles_labels()
    # Filter out the black mean line to ensure it is prominent
    ax3.legend(handles[-len(df['File'].unique())-1:], labels[-len(df['File'].unique())-1:], title='File')
    
    print("Saving Decoding Speedup plots to tunstall_decoding_speedup_by_cl.png")
    fig3.savefig('tunstall_decoding_speedup_by_cl.png', bbox_inches='tight', dpi=300)
    plt.close(fig3)
    
    # --- Plot 4: Overall Average Decoding Time per Decoder Across All Code Lengths ---
    df_avg_time = df.melt(
        id_vars=['File', 'CodeLength'], 
        value_vars=['SerialTime_ms', 'ParallelTime_ms'],
        var_name='Decoder', 
        value_name='Time_ms'
    ).groupby('Decoder')['Time_ms'].mean().reset_index()

    fig4, ax4 = plt.subplots(figsize=(7, 6))

    # Rename for clearer legend in the plot
    df_avg_time['Decoder'] = df_avg_time['Decoder'].str.replace('Time_ms', ' Decoder')

    sns.barplot(
        x='Decoder', 
        y='Time_ms', 
        data=df_avg_time, 
        palette={'Serial Decoder': '#fb923c', 'Parallel Decoder': '#059669'},
        ax=ax4
    )

    avg_speedup = df_avg_time.loc[df_avg_time['Decoder'] == 'Serial Decoder', 'Time_ms'].iloc[0] / df_avg_time.loc[df_avg_time['Decoder'] == 'Parallel Decoder', 'Time_ms'].iloc[0]

    ax4.set_title(f'Overall Average Decoding Time (Speedup: x{avg_speedup:.1f})', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Decoder Type', fontsize=12)
    ax4.set_ylabel('Average Decoding Time (ms)', fontsize=12)
    
    # Add labels on top of the bars
    for index, row in df_avg_time.iterrows():
        ax4.text(
            index, 
            row.Time_ms + 10, 
            f'{row.Time_ms:.1f} ms', 
            color='black', 
            ha='center', 
            fontweight='bold'
        )
    
    print("Saving Overall Average Time plots to tunstall_overall_average_time.png")
    fig4.savefig('tunstall_overall_average_time.png', bbox_inches='tight', dpi=300)
    plt.close(fig4)




def main_visualize():
    """
    Main function to execute the visualization by reading real files.
    """
    try:
        df = read_and_process_all_results(directory='.')
        if not df.empty:
            generate_plots(df)
        else:
            print("Visualization failed: Could not load data from files.", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred during visualization: {e}", file=sys.stderr)

if __name__ == '__main__':
    main_visualize()