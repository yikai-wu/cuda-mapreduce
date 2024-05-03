import numpy as np
from numba import cuda, jit, float32
import time
from functools import reduce

@cuda.jit
def map_kernel(A, B, mapped_values):
    # Thread and block indices
    i, j = cuda.grid(2)
    if i < A.shape[0] and j < B.shape[1]:
        for k in range(B.shape[0]):
            mapped_values[i, j, k, 0] = A[i, k]
            mapped_values[i, j, k, 1] = B[k, j]

@cuda.jit
def reduce_kernel(mapped_values, C):
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        sum = 0
        for k in range(mapped_values.shape[2]):
            sum += mapped_values[i, j, k, 0] * mapped_values[i, j, k, 1]
        C[i, j] = sum

def matrix_multiply_gpu(A, B):
    gpu_start = time.time()
    
    # Convert arrays to device arrays
    A_device = cuda.to_device(A)
    B_device = cuda.to_device(B)
    mapped_values = cuda.device_array((A.shape[0], B.shape[1], A.shape[1], 2), dtype=np.float32)
    
    # Dimensions for mapping
    threadsperblock = (16, 16)
    blockspergrid_x = (A.shape[0] + (threadsperblock[0] - 1)) // threadsperblock[0]
    blockspergrid_y = (B.shape[1] + (threadsperblock[1] - 1)) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    kernel_start=time.time()
    # Run map kernel
    map_kernel[blockspergrid, threadsperblock](A_device, B_device, mapped_values)
    
    # Result matrix
    C_device = cuda.device_array((A.shape[0], B.shape[1]), dtype=np.float32)
    
    # Run reduce kernel
    reduce_kernel[blockspergrid, threadsperblock](mapped_values, C_device)
    kernel_time=time.time()-kernel_start
    
    # Copy result back to host
    C = C_device.copy_to_host()
    gpu_time = time.time() - gpu_start
    
    return C, gpu_time, kernel_time


def matrix_multiply_cpu(A, B):
    cpu_start=time.time()
    # Map function to prepare the (key, value) pairs for matrix multiplication
    def map_matrix_mult(A, B):
        mapped_values = []
        n = len(B)  # Assuming B is well-formed as an n x p matrix
        for i, row in enumerate(A):
            for j in range(len(B[0])):
                for k in range(n):
                    mapped_values.append(((i, j), (row[k], B[k][j])))
        return mapped_values

    # Reduce function to compute the matrix product from mapped values
    def reduce_matrix_mult(mapped_values):
        def reducer(accumulator, value):
            key, numbers = value
            result = accumulator.get(key, 0) + numbers[0] * numbers[1]
            accumulator[key] = result
            return accumulator
        
        return reduce(reducer, mapped_values, {})

    # Perform the map and reduce operations
    # mapped_values = map_matrix_mult(A, B)
    # reduced_result = reduce_matrix_mult(mapped_values)

    # # Format the result into the matrix form
    # m = len(A)
    # p = len(B[0])
    # C = [[reduced_result.get((i, j), 0) for j in range(p)] for i in range(m)]
    C=np.dot(A, B)
    cpu_time=time.time()-cpu_start
    return C, cpu_time

def read_matrices_from_files(file_A='matrix_A_lg.npy', file_B='matrix_B_lg.npy'):
    A = np.load(file_A)
    B = np.load(file_B)
    return A, B

if __name__ == "__main__":
    A, B = read_matrices_from_files()
    print(A.shape)
    print(B.shape)


    # Perform matrix multiplication on the CPU
    total_cpu_time=0.0
    total_gpu_time=0.0
    total_kernal_time=0.0
    for i in range(0,4):
        C_cpu, cpu_time = matrix_multiply_cpu(A, B)
        total_cpu_time+=cpu_time
    for i in range(0,4):
        C, gpu_time, kernel_time = matrix_multiply_gpu(A, B)
        total_gpu_time+=gpu_time
        total_kernel_time+=kernel_time

    print(f"CPU time: {total_cpu_time}s, GPU time: {total_gpu_time}s, Kernel time: {total_kernel_time}s")
    # print("CPU results:")
    # print("C matrix CPU:", C_cpu)
    # print("GPU results:")
    # print("C matrix GPU:", C)