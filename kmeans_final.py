import numpy as np
import warnings
import numba
import time
from sklearn.cluster import KMeans
from numba import cuda
import argparse

@cuda.jit
def assign_labels(data, centroids, labels):
    idx = cuda.grid(1)
    if idx < data.shape[0]:
        min_dist = np.inf
        for i in range(centroids.shape[0]):
            dist = 0.0
            for j in range(data.shape[1]):
                temp = data[idx, j] - centroids[i, j]
                dist += temp * temp
            if dist < min_dist:
                min_dist = dist
                labels[idx] = i

@cuda.jit
def update_centroids(data, centroids, labels, counts):
    idx = cuda.grid(1)
    if idx < data.shape[0]:
        label = labels[idx]
        for j in range(data.shape[1]):
            cuda.atomic.add(centroids, (label, j), data[idx, j])
        cuda.atomic.add(counts, label, 1)

@cuda.jit
def finalize_centroids(centroids, counts):
    idx = cuda.grid(1)
    num_centroids = centroids.shape[0]
    num_features = centroids.shape[1]
    total_elements = num_centroids * num_features
    
    if idx < total_elements:
        centroid_idx = idx // num_features  # Determine which centroid
        feature_idx = idx % num_features    # Determine which feature of the centroid
        
        if counts[centroid_idx] > 0:
            centroids[centroid_idx, feature_idx] /= counts[centroid_idx]

@cuda.jit(fastmath=True)
def fill_zeros(arr):
    idx = cuda.grid(1)
    if idx < arr.size:
        arr[idx] = 0

def gpu_kmeans(data, k=3, max_iter=300):
    start_time=time.time()
    
    threads_per_block = 128
    num_points = data.shape[0]
    blocks_per_grid_points = (num_points + threads_per_block - 1) // threads_per_block
    
    blocks_per_grid_centroids = (k * data.shape[1] + threads_per_block - 1) // threads_per_block
    blocks_per_grid_counts = (k + threads_per_block - 1) // threads_per_block

    data_device = cuda.to_device(data)
    centroids = np.random.rand(k, data.shape[1]).astype(np.float32)
    centroids_device = cuda.to_device(centroids)
    labels = np.zeros(num_points, dtype=np.int32)
    labels_device = cuda.to_device(labels)
    counts = np.zeros(k, dtype=np.int32)
    counts_device = cuda.to_device(counts)

    cuda.synchronize()
    kernel_start_time=time.time()
    for _ in range(max_iter):
        assign_labels[blocks_per_grid_points, threads_per_block](data_device, centroids_device, labels_device)
        fill_zeros[blocks_per_grid_centroids, threads_per_block](centroids_device.ravel())
        fill_zeros[blocks_per_grid_counts, threads_per_block](counts_device)
        update_centroids[blocks_per_grid_points, threads_per_block](data_device, centroids_device, labels_device, counts_device)
        finalize_centroids[blocks_per_grid_centroids, threads_per_block](centroids_device, counts_device)
    kernel_end_time=time.time()
    cuda.synchronize()
    kernel_time=kernel_end_time-kernel_start_time
        
    centroids = centroids_device.copy_to_host()
    labels = labels_device.copy_to_host()
    end_time=time.time()
    gpu_time=end_time-start_time
    return centroids, labels, gpu_time, kernel_time

def cpu_kmeans(data, k=5, max_iter=300):
    start_time = time.time()
    kmeans = KMeans(n_clusters=k, max_iter=max_iter, n_init=10)
    kmeans.fit(data)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    end_time = time.time()
    cpu_time = end_time-start_time
    return centroids, labels, cpu_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--num_features', type=int, default=2000)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--n_clusters', type=int, default=5)
    parser.add_argument('--max_iter', type=int, default=300)
    parser.add_argument('--warning', type=bool, default=False)
    args = parser.parse_args()
    if not args.warning:
        warnings.filterwarnings('ignore', category=numba.NumbaPerformanceWarning)
        
    data = np.load("num_data.npy", dtype=np.float32)
    # data = np.random.rand(args.num_samples, args.num_features).astype(np.float32)
    cpu_centroids, cpu_labels, cpu_time = cpu_kmeans(data, k=args.n_clusters, max_iter=args.max_iter)
    gpu_centroids, gpu_labels, gpu_time, kernel_time = gpu_kmeans(data, k=args.n_clusters, max_iter=args.max_iter)
    print(f"CPU time: {cpu_time}s, GPU time: {gpu_time}s, Kernel time: {kernel_time}s")
    print(f"CPU time: {cpu_time}s, GPU time: {gpu_time}s, Kernel time: {kernel_time}s")
    print("CPU results:")
    print("CPU Centroids:", cpu_centroids)
    print("CPU Labels:", cpu_labels)
    print("GPU results:")
    print("GPU Centroids:", gpu_centroids)
    print("GPU Labels:", gpu_labels)