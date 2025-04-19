from mpi4py import MPI
import numpy as np
import time
import sys

def distributed_memory_matrix_multiplication(A, B, comm):
    """
    Parallel matrix multiplication using distributed memory model with MPI.
    A: full matrix (m x n) - only available at root process
    B: full matrix (n x p) - only available at root process
    comm: MPI communicator
    Returns: result matrix C (m x p) - only available at root process
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Only root process (rank 0) has input data
    if rank == 0:
        m, n = A.shape
        n, p = B.shape
        
        # Check dimensions
        if A.shape[1] != B.shape[0]:
            raise ValueError("Matrix dimensions incompatible for multiplication")
    else:
        m = n = p = None
    
    # Broadcast matrix dimensions
    m, n, p = comm.bcast((m, n, p), root=0)
    
    # Broadcast matrix B to all processes
    if rank == 0:
        B_full = B
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
        comm.Scatterv([A, send_counts, displacements, MPI.DOUBLE], local_A, root=0)
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
        C = np.empty((m, p))
        
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
        
        comm.Gatherv(local_C, [C, recv_counts, displacements, MPI.DOUBLE], root=0)
    else:
        comm.Gatherv(local_C, None, root=0)
    
    # Only root process returns result
    if rank == 0:
        return C
    return None

def test_distributed_memory(sizes):
    """
    Test distributed memory parallel algorithm with different matrix sizes.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    times = []
    
    for size in sizes:
        # Only root process creates matrices
        if rank == 0:
            A = np.random.random((size, size))
            B = np.random.random((size, size))
        else:
            A = None
            B = None
        
        # Synchronize all processes before timing
        comm.Barrier()
        
        # Measure execution time
        start_time = MPI.Wtime()
        C = distributed_memory_matrix_multiplication(A, B, comm)
        end_time = MPI.Wtime()
        
        if rank == 0:
            # Verify correctness with numpy
            numpy_C = np.matmul(A, B)
            assert np.allclose(C, numpy_C), f"Result incorrect for size {size}"
            
            elapsed_time = end_time - start_time
            times.append(elapsed_time)
            print(f"Matrix size: {size}x{size}, Parallel time (distributed memory): {elapsed_time:.4f} seconds")
    
    return times

if __name__ == "__main__":
    # Run with mpiexec -n <num_processes> python distributed_matrix_multiplication.py
    sizes = [100, 200, 500]
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    distributed_times = test_distributed_memory(sizes)
    
    # Only root process prints results
    if rank == 0:
        print("Completed distributed memory test")