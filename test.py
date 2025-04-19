import numpy as np
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os
import pandas as pd
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

# Sequential matrix multiplication
def matrix_multiply_sequential(A, B):
    """Standard sequential matrix multiplication implementation"""
    m = len(A)
    n = len(B)
    p = len(B[0]) if isinstance(B[0], list) else len(B)
    
    C = [[0 for _ in range(p)] for _ in range(m)]
    
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
                
    return C

# NumPy-based sequential multiplication (for efficient baseline)
def matrix_multiply_numpy(A, B):
    """NumPy-based matrix multiplication (efficient sequential implementation)"""
    return np.dot(np.array(A), np.array(B))

# Shared memory model - multiprocessing approach
def multiply_row_by_matrix(args):
    """Helper function for shared memory parallelization"""
    row_idx, row, matrix = args
    result = np.zeros(len(matrix[0]))
    for j in range(len(matrix[0])):
        for k in range(len(row)):
            result[j] += row[k] * matrix[k][j]
    return row_idx, result

def matrix_multiply_shared_memory(A, B, num_processes=4):
    """Matrix multiplication using multiprocessing (shared memory model)"""
    m = len(A)
    p = len(B[0]) if isinstance(B[0], list) else len(B)
    
    # Convert to numpy arrays if not already
    if not isinstance(A, np.ndarray):
        A = np.array(A)
    if not isinstance(B, np.ndarray):
        B = np.array(B)
    
    # Prepare result matrix
    C = np.zeros((m, p))
    
    # Create process pool
    with Pool(processes=num_processes) as pool:
        # Create tasks: each task multiplies one row of A by B
        tasks = [(i, A[i], B) for i in range(m)]
        
        # Execute tasks in parallel and collect results
        results = pool.map(multiply_row_by_matrix, tasks)
        
        # Assemble results into the final matrix
        for row_idx, row_result in results:
            C[row_idx] = row_result
    
    return C

# Function to simulate MPI-like behavior for distributed memory model
# Note: In a real environment, this would use mpi4py
class MPISimulator:
    """Simulate MPI-like behavior for demonstration purposes"""
    def __init__(self, num_processes):
        self.num_processes = num_processes

    def scatter_gather_multiply(self, A, B):
        """Simulate scatter-gather matrix multiplication with distributed memory"""
        start_time = time.time()
        m = len(A)
        
        # Determine chunk size for each process
        chunk_size = m // self.num_processes
        if chunk_size == 0:
            chunk_size = 1
        
        results = []
        
        # Simulate distributing work among processes
        for proc_id in range(self.num_processes):
            start_row = proc_id * chunk_size
            end_row = start_row + chunk_size if proc_id < self.num_processes - 1 else m
            
            # Process local chunk (simulate work done by each process)
            local_result = np.dot(A[start_row:end_row], B)
            results.append(local_result)
        
        # Gather results (simulate MPI gather operation)
        C = np.vstack(results)
        
        end_time = time.time()
        return C, end_time - start_time

def matrix_multiply_distributed(A, B, num_processes=4):
    """Simulate distributed memory matrix multiplication"""
    if not isinstance(A, np.ndarray):
        A = np.array(A)
    if not isinstance(B, np.ndarray):
        B = np.array(B)
    
    mpi_sim = MPISimulator(num_processes)
    result, _ = mpi_sim.scatter_gather_multiply(A, B)
    return result

# Evaluation functions
def verify_results(A, B):
    """Verify that all implementations produce the same result"""
    # Reference result using NumPy
    ref_result = np.dot(np.array(A), np.array(B))
    
    # Sequential implementation result
    seq_result = np.array(matrix_multiply_sequential(A, B))
    seq_diff = np.linalg.norm(ref_result - seq_result)
    
    # Shared memory implementation result
    shared_result = matrix_multiply_shared_memory(A, B, num_processes=2)
    shared_diff = np.linalg.norm(ref_result - shared_result)
    
    # Distributed memory implementation result
    dist_result = matrix_multiply_distributed(A, B, num_processes=2)
    dist_diff = np.linalg.norm(ref_result - dist_result)
    
    print(f"Sequential vs NumPy difference: {seq_diff}")
    print(f"Shared memory vs NumPy difference: {shared_diff}")
    print(f"Distributed memory vs NumPy difference: {dist_diff}")
    
    return seq_diff < 1e-10 and shared_diff < 1e-10 and dist_diff < 1e-10

def benchmark_single_run(size, num_processes_list=[1, 2, 4, 8]):
    """Benchmark all implementations for a single matrix size"""
    # Generate random matrices
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    
    results = {'Size': size}
    
    # Sequential NumPy (reference)
    start_time = time.time()
    np.dot(A, B)
    seq_time = time.time() - start_time
    results['Sequential'] = seq_time
    
    # Shared Memory with different process counts
    for num_proc in num_processes_list:
        if num_proc <= os.cpu_count():  # Only run if we have enough cores
            start_time = time.time()
            matrix_multiply_shared_memory(A, B, num_processes=num_proc)
            shared_time = time.time() - start_time
            results[f'Shared-{num_proc}'] = shared_time
    
    # Distributed Memory with different process counts
    for num_proc in num_processes_list:
        if num_proc <= os.cpu_count():  # Only run if we have enough cores
            start_time = time.time()
            matrix_multiply_distributed(A, B, num_processes=num_proc)
            dist_time = time.time() - start_time
            results[f'Distributed-{num_proc}'] = dist_time
    
    return results

def run_benchmarks(sizes=[10, 50, 100, 200, 500, 1000], num_processes_list=[1, 2, 4, 8]):
    """Run benchmarks for all matrix sizes and implementations"""
    all_results = []
    
    for size in sizes:
        print(f"\nBenchmarking matrices of size {size}x{size}")
        result = benchmark_single_run(size, num_processes_list)
        all_results.append(result)
        
        # Print current results
        print(f"Sequential: {result['Sequential']:.4f} seconds")
        for k, v in result.items():
            if k != 'Size' and k != 'Sequential':
                speedup = result['Sequential'] / v
                print(f"{k}: {v:.4f} seconds (Speedup: {speedup:.2f}x)")
    
    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame(all_results)
    return results_df

def calculate_speedup(results_df):
    """Calculate speedup relative to sequential execution"""
    speedup_df = pd.DataFrame({'Size': results_df['Size']})
    
    for column in results_df.columns:
        if column != 'Size' and column != 'Sequential':
            speedup_df[column] = results_df['Sequential'] / results_df[column]
    
    return speedup_df

def plot_execution_times(results_df):
    """Plot execution times for all implementations"""
    # Reshape the DataFrame for plotting
    plot_df = pd.melt(results_df, id_vars=['Size'], 
                       var_name='Implementation', value_name='Time (seconds)')
    
    plt.figure(figsize=(12, 8))
    
    # Use log scale for better visibility
    sns.lineplot(data=plot_df, x='Size', y='Time (seconds)', 
                 hue='Implementation', marker='o', linewidth=2.5)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.7)
    
    plt.title('Matrix Multiplication Performance Comparison', fontsize=16)
    plt.xlabel('Matrix Size (n × n)', fontsize=14)
    plt.ylabel('Execution Time (seconds)', fontsize=14)
    
    # Format ticks to show actual values instead of powers
    for axis in [plt.gca().xaxis, plt.gca().yaxis]:
        axis.set_major_formatter(ScalarFormatter())
    
    plt.legend(title='Implementation', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('matrix_multiplication_time_comparison.png', dpi=300)
    plt.close()

def plot_speedup(speedup_df):
    """Plot speedup for all parallel implementations"""
    # Reshape the DataFrame for plotting
    plot_df = pd.melt(speedup_df, id_vars=['Size'], 
                       var_name='Implementation', value_name='Speedup')
    
    plt.figure(figsize=(12, 8))
    
    sns.lineplot(data=plot_df, x='Size', y='Speedup', 
                 hue='Implementation', marker='o', linewidth=2.5)
    
    plt.xscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.7)
    
    # Add a horizontal line at y=1 (baseline - no speedup)
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='Baseline (Sequential)')
    
    # Add ideal speedup lines for reference
    max_speedup = plot_df['Speedup'].max() * 1.2
    x = np.array(speedup_df['Size'])
    for p in [2, 4, 8]:
        if max_speedup >= p:
            plt.axhline(y=p, color='gray', linestyle=':', alpha=0.5)
            plt.text(x[-1]*1.05, p, f'Ideal {p}x', va='center', alpha=0.7)
    
    plt.title('Speedup of Parallel Matrix Multiplication Implementations', fontsize=16)
    plt.xlabel('Matrix Size (n × n)', fontsize=14)
    plt.ylabel('Speedup (Relative to Sequential)', fontsize=14)
    
    # Format x-axis ticks to show actual values instead of powers
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    
    plt.legend(title='Implementation', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('matrix_multiplication_speedup_comparison.png', dpi=300)
    plt.close()

def plot_efficiency(speedup_df):
    """Plot parallel efficiency for all implementations"""
    efficiency_df = pd.DataFrame({'Size': speedup_df['Size']})
    
    # Calculate efficiency = speedup / number of processes
    for column in speedup_df.columns:
        if column != 'Size':
            # Extract number of processes from column name
            if 'Shared' in column or 'Distributed' in column:
                num_procs = int(column.split('-')[1])
                efficiency_df[column] = speedup_df[column] / num_procs
    
    # Reshape the DataFrame for plotting
    plot_df = pd.melt(efficiency_df, id_vars=['Size'], 
                       var_name='Implementation', value_name='Efficiency')
    
    plt.figure(figsize=(12, 8))
    
    sns.lineplot(data=plot_df, x='Size', y='Efficiency', 
                 hue='Implementation', marker='o', linewidth=2.5)
    
    plt.xscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.7)
    
    # Add a horizontal line at y=1 (perfect efficiency)
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='Perfect Efficiency')
    
    plt.title('Efficiency of Parallel Matrix Multiplication Implementations', fontsize=16)
    plt.xlabel('Matrix Size (n × n)', fontsize=14)
    plt.ylabel('Parallel Efficiency (Speedup / # Processes)', fontsize=14)
    
    # Format x-axis ticks to show actual values instead of powers
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    
    plt.legend(title='Implementation', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('matrix_multiplication_efficiency_comparison.png', dpi=300)
    plt.close()

def create_comparison_table(results_df, speedup_df):
    """Create a comparison table with execution times and speedups"""
    # Create a new DataFrame for the table
    table_df = pd.DataFrame({'Size': results_df['Size']})
    
    # Add sequential times
    table_df['Sequential (s)'] = results_df['Sequential']
    
    # Add times and speedups for parallel implementations
    for column in results_df.columns:
        if column != 'Size' and column != 'Sequential':
            table_df[f'{column} (s)'] = results_df[column]
            table_df[f'{column} Speedup'] = speedup_df[column]
    
    return table_df

def combined_visualization(results_df):
    """Create a combined visualization with multiple subplots"""
    speedup_df = calculate_speedup(results_df)
    
    # Create a figure with 2 subplots
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    
    # Reshape the DataFrame for plotting
    time_df = pd.melt(results_df, id_vars=['Size'], 
                       var_name='Implementation', value_name='Time (seconds)')
    
    speedup_plot_df = pd.melt(speedup_df, id_vars=['Size'], 
                              var_name='Implementation', value_name='Speedup')
    
    # Plot 1: Execution Time
    sns.lineplot(data=time_df, x='Size', y='Time (seconds)', 
                 hue='Implementation', marker='o', ax=axs[0])
    
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].grid(True, which="both", ls="--", alpha=0.7)
    axs[0].set_title('Execution Time Comparison', fontsize=14)
    axs[0].set_xlabel('Matrix Size (n × n)', fontsize=12)
    axs[0].set_ylabel('Time (seconds)', fontsize=12)
    
    # Plot 2: Speedup
    sns.lineplot(data=speedup_plot_df, x='Size', y='Speedup', 
                 hue='Implementation', marker='o', ax=axs[1])
    
    axs[1].set_xscale('log')
    axs[1].grid(True, which="both", ls="--", alpha=0.7)
    axs[1].axhline(y=1, color='r', linestyle='--', alpha=0.7)
    
    # Add ideal speedup lines
    max_speedup = speedup_plot_df['Speedup'].max() * 1.2
    for p in [2, 4, 8]:
        if max_speedup >= p:
            axs[1].axhline(y=p, color='gray', linestyle=':', alpha=0.5)
            axs[1].text(speedup_df['Size'].iloc[-1]*1.05, p, f'Ideal {p}x', va='center', alpha=0.7)
    
    axs[1].set_title('Speedup Comparison', fontsize=14)
    axs[1].set_xlabel('Matrix Size (n × n)', fontsize=12)
    axs[1].set_ylabel('Speedup (vs Sequential)', fontsize=12)
    
    # Format ticks
    for ax in axs:
        ax.xaxis.set_major_formatter(ScalarFormatter())
    
    plt.suptitle('Matrix Multiplication Performance Analysis', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig('matrix_multiplication_combined_analysis.png', dpi=300)
    plt.close()

def main():
    # Verify that implementations produce the same results
    print("Verifying implementation correctness...")
    A_small = np.random.rand(10, 10)
    B_small = np.random.rand(10, 10)
    if verify_results(A_small, B_small):
        print("All implementations produce correct results.\n")
    else:
        print("Warning: Implementation discrepancies detected!\n")
    
    # Get the number of available CPU cores
    cpu_count = os.cpu_count()
    print(f"Running on a machine with {cpu_count} CPU cores")
    
    # Determine process counts based on available cores
    if cpu_count >= 8:
        process_counts = [1, 2, 4, 8]
    elif cpu_count >= 4:
        process_counts = [1, 2, 4]
    else:
        process_counts = [1, 2]
    
    # Run benchmarks
    print("\nRunning benchmarks...")
    # Use smaller sizes for demonstration purposes
    # In a real benchmark, you might use larger sizes
    sizes = [10, 50, 100, 200, 500]
    
    # For very capable machines, add 1000
    if cpu_count >= 8:
        sizes.append(1000)
    
    results_df = run_benchmarks(sizes=sizes, num_processes_list=process_counts)
    
    # Calculate speedups
    speedup_df = calculate_speedup(results_df)
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_execution_times(results_df)
    plot_speedup(speedup_df)
    plot_efficiency(speedup_df)
    combined_visualization(results_df)
    
    # Create comparison table
    table_df = create_comparison_table(results_df, speedup_df)
    print("\nPerformance Comparison Table:")
    print(table_df.to_string(index=False))
    
    print("\nAnalysis complete. Visualization files saved.")

if __name__ == "__main__":
    main()