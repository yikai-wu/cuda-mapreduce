import numpy as np
from numba import cuda, jit, float32
import time
import math

def matrix_multiply_cpu(A, B):
    cpu_start=time.time()
   
    C=np.dot(A, B)
    cpu_time=time.time()-cpu_start
    return C, cpu_time

TPB = 32

@cuda.jit
def gpu_matrix_mult(A, B, C):
    """ Perform matrix multiplication of C = A * B """
    x, y = cuda.grid(2)
    if x >= C.shape[0] or y >= C.shape[1]:
        return

    tmp = float(0)
    for k in range(A.shape[1]):
        tmp += A[x, k] * B[k, y]

    C[x, y] = tmp

def matrix_multiply_gpu(A, B):
    """ Host code to setup the computation on the GPU """
    start_time=time.time()
    stream1 = cuda.stream()
    stream2 = cuda.stream()
    stream3 = cuda.stream()

    A_global_mem = cuda.to_device(A, stream=stream1)
    B_global_mem = cuda.to_device(B, stream=stream2)
    C_global_mem = cuda.device_array((A.shape[0], B.shape[1]), dtype=np.float32)

    threadsperblock = (TPB, TPB)
    blockspergrid_x = math.ceil(A.shape[0] / TPB)
    blockspergrid_y = math.ceil(B.shape[1] / TPB)
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    kernel_start_time=time.time()
    gpu_matrix_mult[blockspergrid, threadsperblock, stream3](A_global_mem, B_global_mem, C_global_mem)
    kernel_time=time.time()-kernel_start_time

    # Wait for all streams to complete their tasks
    stream1.synchronize()
    stream2.synchronize()
    stream3.synchronize()

    C = C_global_mem.copy_to_host()
    gpu_time = time.time()-start_time
    return C, gpu_time, kernel_time


