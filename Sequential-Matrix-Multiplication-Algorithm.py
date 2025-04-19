import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def sequential_matrix_multiplication(A, B):
    """
    Sequential matrix multiplication.
    A: matrix (m x n)
    B: matrix (n x p)
    Returns: result matrix C (m x p)
    """
    if A.shape[1] != B.shape[0]:
        raise ValueError("Matrix dimensions incompatible for multiplication")
    
    m, n = A.shape
    n, p = B.shape
    
    C = np.zeros((m, p))
    
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
                
    return C

def test_sequential(sizes):
    """
    Test sequential algorithm with different matrix sizes.
    """
    times = []
    
    for size in sizes:
        # Create random matrices
        A = np.random.random((size, size))
        B = np.random.random((size, size))
        
        # Measure execution time
        start_time = time.time()
        C = sequential_matrix_multiplication(A, B)
        end_time = time.time()
        
        # Verify correctness with numpy
        numpy_C = np.matmul(A, B)
        assert np.allclose(C, numpy_C), f"Result incorrect for size {size}"
        
        elapsed_time = end_time - start_time
        times.append(elapsed_time)
        print(f"Matrix size: {size}x{size}, Sequential time: {elapsed_time:.4f} seconds")
    
    return times

if __name__ == "__main__":
    sizes = [100, 200, 500]
    sequential_times = test_sequential(sizes)