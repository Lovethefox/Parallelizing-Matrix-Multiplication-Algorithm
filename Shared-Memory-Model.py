import numpy as np
import time
import multiprocessing as mp
from functools import partial

def parallel_row_multiplication(A, B, row_idx, result_queue):
    """
    Compute one row of the result matrix C.
    A: matrix (m x n)
    B: matrix (n x p)
    row_idx: row index to compute
    result_queue: queue to store result
    """
    n, p = B.shape
    row_result = np.zeros(p)
    
    for j in range(p):
        for k in range(n):
            row_result[j] += A[row_idx][k] * B[k][j]
    
    # Put result in queue
    result_queue.put((row_idx, row_result))

def shared_memory_matrix_multiplication(A, B, num_processes=None):
    """
    Parallel matrix multiplication using shared memory model.
    A: matrix (m x n)
    B: matrix (n x p)
    num_processes: number of processes (default is number of CPU cores)
    Returns: result matrix C (m x p)
    """
    if A.shape[1] != B.shape[0]:
        raise ValueError("Matrix dimensions incompatible for multiplication")
    
    m = A.shape[0]
    p = B.shape[1]
    
    # Use all CPU cores if not specified
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    # Create result queue
    result_queue = mp.Queue()
    
    # Create process pool
    pool = mp.Pool(processes=num_processes)
    
    # Create task for each row of result matrix
    row_func = partial(parallel_row_multiplication, A, B)
    
    # Distribute tasks
    for i in range(m):
        pool.apply_async(row_func, args=(i, result_queue))
    
    # Close pool and wait for all processes to complete
    pool.close()
    pool.join()
    
    # Collect results
    C = np.zeros((m, p))
    for _ in range(m):
        row_idx, row_result = result_queue.get()
        C[row_idx] = row_result
    
    return C

def test_shared_memory(sizes):
    """
    Test shared memory parallel algorithm with different matrix sizes.
    """
    times = []
    
    for size in sizes:
        # Create random matrices
        A = np.random.random((size, size))
        B = np.random.random((size, size))
        
        # Measure execution time
        start_time = time.time()
        C = shared_memory_matrix_multiplication(A, B)
        end_time = time.time()
        
        # Verify correctness with numpy
        numpy_C = np.matmul(A, B)
        assert np.allclose(C, numpy_C), f"Result incorrect for size {size}"
        
        elapsed_time = end_time - start_time
        times.append(elapsed_time)
        print(f"Matrix size: {size}x{size}, Parallel time (shared memory): {elapsed_time:.4f} seconds")
    
    return times

if __name__ == "__main__":
    sizes = [100, 200, 500]
    shared_memory_times = test_shared_memory(sizes)