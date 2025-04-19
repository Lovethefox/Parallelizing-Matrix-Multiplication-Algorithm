import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import time
import multiprocessing as mp
import pandas as pd
from mpi4py import MPI

# Import functions from previous files
from sequential_matrix_multiplication import sequential_matrix_multiplication
from shared_memory_parallel import shared_memory_matrix_multiplication
# Note: Cannot directly import from distributed_memory_parallel as it needs to run with mpiexec

def run_comparison(sizes=[100, 200, 500, 1000], verify=True):
    """
    Compare the performance of three matrix multiplication methods.
    """
    sequential_times = []
    shared_memory_times = []
    
    # Only root process collects times for distributed memory method
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print(f"Number of MPI processes: {size}")
        print(f"Number of CPU cores: {mp.cpu_count()}")
        print("\n=== Matrix Multiplication Performance Comparison ===")
    
    # Execute and time for each matrix size
    for matrix_size in sizes:
        if rank == 0:
            print(f"\nMatrix size: {matrix_size}x{matrix_size}")
            
            # Create random matrices
            A = np.random.random((matrix_size, matrix_size))
            B = np.random.random((matrix_size, matrix_size))
            
            # Compute reference result for verification
            if verify:
                expected_C = np.matmul(A, B)
            
            # Sequential method
            start_time = time.time()
            C_sequential = sequential_matrix_multiplication(A, B)
            seq_time = time.time() - start_time
            sequential_times.append(seq_time)
            print(f"  Sequential time: {seq_time:.4f} seconds")
            
            if verify:
                assert np.allclose(C_sequential, expected_C), "Sequential result incorrect!"
            
            # Shared memory method
            start_time = time.time()
            C_shared = shared_memory_matrix_multiplication(A, B)
            shared_time = time.time() - start_time
            shared_memory_times.append(shared_time)
            print(f"  Parallel time (shared memory): {shared_time:.4f} seconds")
            
            if verify:
                assert np.allclose(C_shared, expected_C), "Shared memory result incorrect!"
                
            # Save input matrices for distribution
            input_A = A
            input_B = B
        else:
            input_A = None
            input_B = None
            expected_C = None
        
        # Synchronize all processes
        comm.Barrier()
        
        # Distributed memory method (runs on all processes)
        start_time = MPI.Wtime()
        
        # Broadcast matrix dimensions
        if rank == 0:
            m, n = input_A.shape
            n, p = input_B.shape
        else:
            m = n = p = None
        
        m, n, p = comm.bcast((m, n, p), root=0)
        
        # Broadcast matrix B to all processes
        if rank == 0:
            B_full = input_B
        else:
            B_full = np.empty((n, p))
        
        comm.Bcast(B_full, root=0)
        
        # Calculate rows per process
        rows_per_process = m // size
        remainder = m % size
        
        # Determine start and end rows for each process
        if rank < remainder:
            start_row = rank * (rows_per_process + 1)
            num_rows = rows_per_process + 1
        else:
            start_row = rank * rows_per_process + remainder
            num_rows = rows_per_process
        
        # Distribute rows of matrix A
        if rank == 0:
            send_counts = np.zeros(size, dtype=int)
            displacements = np.zeros(size, dtype=int)
            
            for i in range(size):
                if i < remainder:
                    send_counts[i] = (rows_per_process + 1) * n
                    displacements[i] = i * (rows_per_process + 1) * n
                else:
                    send_counts[i] = rows_per_process * n
                    displacements[i] = (remainder * (rows_per_process + 1) + 
                                       (i - remainder) * rows_per_process) * n
                    
            local_A = np.empty((num_rows, n))
            comm.Scatterv([input_A, send_counts, displacements, MPI.DOUBLE], local_A, root=0)
        else:
            local_A = np.empty((num_rows, n))
            comm.Scatterv(None, local_A, root=0)
        
        # Compute portion of result matrix
        local_C = np.zeros((num_rows, p))
        
        for i in range(num_rows):
            for j in range(p):
                for k in range(n):
                    local_C[i][j] += local_A[i][k] * B_full[k][j]
        
        # Gather results from all processes
        if rank == 0:
            C_distributed = np.empty((m, p))
            
            # Prepare receive information
            recv_counts = np.zeros(size, dtype=int)
            displacements = np.zeros(size, dtype=int)
            
            for i in range(size):
                if i < remainder:
                    recv_counts[i] = (rows_per_process + 1) * p
                    displacements[i] = i * (rows_per_process + 1) * p
                else:
                    recv_counts[i] = rows_per_process * p
                    displacements[i] = (remainder * (rows_per_process + 1) + 
                                       (i - remainder) * rows_per_process) * p
            
            comm.Gatherv(local_C, [C_distributed, recv_counts, displacements, MPI.DOUBLE], root=0)
        else:
            comm.Gatherv(local_C, None, root=0)
        
        end_time = MPI.Wtime()
        distributed_time = end_time - start_time
        
        # Only root process prints results and verifies
        if rank == 0:
            print(f"  Parallel time (distributed memory): {distributed_time:.4f} seconds")
            
            if verify:
                assert np.allclose(C_distributed, expected_C), "Distributed memory result incorrect!"
            
            # Calculate speedup
            speedup_shared = seq_time / shared_time
            speedup_distributed = seq_time / distributed_time
            
            print(f"  Speedup (shared memory): {speedup_shared:.2f}x")
            print(f"  Speedup (distributed memory): {speedup_distributed:.2f}x")
    
    # Only root process creates charts
    if rank == 0:
        # Plot execution time chart
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        
        # Create data for time chart
        plt.plot(sizes, sequential_times, 'o-', label='Sequential')
        plt.plot(sizes, shared_memory_times, 's-', label='Parallel (Shared Memory)')
        plt.plot(sizes, [sequential_times[i]/speedup_distributed for i in range(len(sizes))], '^-', label='Parallel (Distributed Memory)')
        
        plt.xlabel('Matrix Size (n x n)')
        plt.ylabel('Execution Time (seconds)')
        plt.title('Execution Time Comparison')
        plt.legend()
        plt.grid(True)
        plt.xscale('log')
        plt.yscale('log')
        
        # Plot speedup chart
        plt.subplot(1, 2, 2)
        
        # Calculate speedup
        speedup_shared_list = [sequential_times[i] / shared_memory_times[i] for i in range(len(sizes))]
        speedup_distributed_list = [sequential_times[i] / (sequential_times[i]/speedup_distributed) for i in range(len(sizes))]
        
        plt.plot(sizes, speedup_shared_list, 's-', label='Shared Memory')
        plt.plot(sizes, speedup_distributed_list, '^-', label='Distributed Memory')
        plt.axhline(y=mp.cpu_count(), color='r', linestyle=':', label=f'Number of CPU cores ({mp.cpu_count()})')
        plt.axhline(y=size, color='g', linestyle=':', label=f'Number of MPI processes ({size})')
        
        plt.xlabel('Matrix Size (n x n)')
        plt.ylabel('Speedup')
        plt.title('Speedup Comparison')
        plt.legend()
        plt.grid(True)
        plt.xscale('log')
        
        plt.tight_layout()
        plt.savefig('matrix_multiplication_performance.png', dpi=300)
        plt.show()
        
        # Create results table
        results = pd.DataFrame({
            'Size': sizes,
            'Sequential (s)': sequential_times,
            'Shared Memory (s)': shared_memory_times,
            'Distributed Memory (s)': [sequential_times[i]/speedup_distributed for i in range(len(sizes))],
            'Speedup (Shared Memory)': speedup_shared_list,
            'Speedup (Distributed Memory)': speedup_distributed_list
        })
        
        print("\n=== Results Table ===")
        print(results.to_string(index=False))
        
        # Save results to CSV file
        results.to_csv('matrix_multiplication_results.csv', index=False)
        
        print("\nResults and chart saved!")

if __name__ == "__main__":
    # Run with command: mpiexec -n <num_processes> python performance_comparison.py
    sizes = [100, 200, 500, 1000]
    run_comparison(sizes)