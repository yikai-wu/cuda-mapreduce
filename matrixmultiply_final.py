import numpy as np
from numba import cuda, jit, float32
import time
import math
import torch

def matrix_multiply_cpu(A, B):
    cpu_start=time.time()
   
    C=np.dot(A, B)
    cpu_time=time.time()-cpu_start
    return C, cpu_time


@cuda.jit
def fast_matmul(A, B, C):
    TPB = 16

    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    

    if x >= C.shape[0] and y >= C.shape[1]:
        return

    tmp = 0.
    for i in range(bpg):
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        cuda.syncthreads()

        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        cuda.syncthreads()

    C[x, y] = tmp

def matrix_multiply_gpu(A, B):
    TPB = 16
    n, k = A.shape
    k, m = B.shape
    start = time.time()
    A_device = cuda.to_device(A)
    B_device = cuda.to_device(B)
    C_device = cuda.device_array((n, m), dtype=np.float32)

    threads_per_block = (TPB, TPB)
    blocks_per_grid_x = math.ceil(m / TPB)
    blocks_per_grid_y = math.ceil(n / TPB)

    cuda.synchronize()
    kernel_start_time = time.time()
    fast_matmul[(blocks_per_grid_y, blocks_per_grid_x), threads_per_block](A_device, B_device, C_device)
    cuda.synchronize()  # Wait for all GPU activity to finish
    kernel_time = time.time()-kernel_start_time

    C = C_device.copy_to_host()
    gpu_time = time.time() - start
    return C, kernel_time, gpu_time

def random_matrix(n):
    A = np.random.default_rng().standard_normal(size=(n,n), dtype='float32')
    B = np.random.default_rng().standard_normal(size=(n,n), dtype='float32')
    return A, B

def matrix_multiply_torch(A, B):
    start_time = time.time()
    device = torch.device("cuda:0")
    A_device = torch.from_numpy(A).to(device)
    B_device = torch.from_numpy(B).to(device)
    kernel_start_time = time.time()
    C_device = torch.matmul(A_device, B_device)
    kernel_time = time.time()-kernel_start_time
    C = C_device.cpu().numpy()
    torch_time = time.time()-start_time
    return C, torch_time, kernel_time

def read_matrices_from_files(file_A='matrix_A_lg.npy', file_B='matrix_B_lg.npy'):
    A = np.load(file_A)
    B = np.load(file_B)
    return A, B

if __name__ == "__main__":
    A, B = read_matrices_from_files()
    print(A.shape)
    print(B.shape)


    # Perform matrix multiplication on the CPU
    cpu_start=time.time()
    total_kernel_time=0.0
    for i in range(0,4):
        C_cpu, cpu_time = matrix_multiply_cpu(A, B)
    total_cpu_time=time.time()-cpu_start
    gpu_start=time.time()
    for i in range(0,4):
        C, gpu_time, kernel_time = matrix_multiply_gpu(A, B)
        total_kernel_time+=kernel_time
    total_gpu_time=time.time-gpu_start

    print(f"CPU time: {total_cpu_time}s, GPU time: {total_gpu_time}s, Kernel time: {total_kernel_time}s")

