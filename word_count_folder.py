import numpy as np
import re
from numba import cuda, types, uint32, int32, jit, njit
from collections import Counter
import string
import time

@cuda.jit(device=True)
def isalpha(c):
    if c >= ord('A') and c <= ord('Z'):
        return True
    elif c >= ord('a') and c <= ord('z'):
        return True
    else:
        return False

@cuda.jit(device=True)
def tolower(c):
    if c >= ord('A') and c <= ord('Z'):
        return c+32
    else:
        return c

@cuda.jit(device=True)
def hash_byte_array(input_array):
    hash_value = np.uint32(5381) 
    for i in range(input_array.size):
        hash_value = ((hash_value << 5) + hash_value) + np.uint32(input_array[i])
    return hash_value

@cuda.jit
def map(text, counts, hash_start, hash_length, hash_size, chunk_size):
    idx = cuda.grid(1)
    if idx*chunk_size >= text.shape[0]:
        return
    chunk_end = (idx+1)*chunk_size
    if chunk_end > text.shape[0]:
        chunk_end = text.shape[0]

    s = -1
    t = idx*chunk_size
    if idx == 0 or (not isalpha(text[t-1])):
        if isalpha(text[t]):
            s = t
    else:
        while (t < chunk_end and isalpha(text[t])):
            t += 1
    while (t < chunk_end):
        if isalpha(text[t]) and (s == -1):
                s = t
        elif (not isalpha(text[t])) and (s != -1):
            word = text[s:t]
            for i in range(len(word)):
                word[i] = tolower(word[i])
            hash_value = hash_byte_array(word) % hash_size
            cuda.atomic.add(counts, hash_value, 1)
            cuda.atomic.exch(hash_start, hash_value, s)
            cuda.atomic.exch(hash_length, hash_value, t-s)
            s = -1
        t += 1
    if s != -1:
        while (t < text.shape[0]) and (isalpha(text[t])):
            t += 1
        word = text[s:t]
        for i in range(len(word)):
            word[i] = tolower(word[i])
        hash_value = hash_byte_array(word) % hash_size
        cuda.atomic.add(counts, hash_value, 1)
        cuda.atomic.exch(hash_start, hash_value, s)
        cuda.atomic.exch(hash_length, hash_value, t-s)
        s = -1

def gpu_word_count(text, hash_size=65536, chunk_size=128):
    start_time=time.time()
    char_array = np.frombuffer(text.encode('utf-8'), dtype=np.byte)

    d_text = cuda.to_device(char_array)

    d_counts = cuda.device_array(hash_size, dtype=np.int32)
    d_hash_start = cuda.device_array(hash_size, dtype=np.int32)
    d_hash_length = cuda.device_array(hash_size, dtype=np.int32)

    threads_per_block = 256
    blocks_per_grid = (len(char_array) + threads_per_block - 1) // threads_per_block

    cuda.synchronize()
    kernel_start_time=time.time()
    map[blocks_per_grid, threads_per_block](d_text, d_counts, d_hash_start, d_hash_length, hash_size, chunk_size)
    cuda.synchronize()
    kernel_end_time=time.time()
    kernel_time = kernel_end_time-kernel_start_time

    char_array = d_text.copy_to_host()
    counts = d_counts.copy_to_host()
    hash_start = d_hash_start.copy_to_host()
    hash_length = d_hash_length.copy_to_host()
    end_time = time.time()
    gpu_time=end_time-start_time
    return char_array, counts, hash_start, hash_length, gpu_time, kernel_time


def cpu_word_count(text):
    start_time = time.time()
    text = text.lower()

    words = re.findall(r'[a-zA-Z]+', text)
    
    word_counts = Counter(words)
    end_time = time.time()
    cpu_time = end_time-start_time
    return word_counts, cpu_time

if __name__ == "__main__":
    folder_path = "data/wikipedia_50GB"
    text = "" 
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                text += file.read() 

    word_counts, cpu_time = cpu_word_count(text)
    char_array, counts, hash_start, hash_length, gpu_time, kernel_time = gpu_word_count(text, hash_size=65536, chunk_size=1)

    print(f"cpu: {cpu_time}s, gpu: {gpu_time}s, kernel: {kernel_time}s")

    print("CPU results:")
    items = word_counts.most_common(100)
    print(items)

    print("GPU results:")
    indices = np.argsort(counts)[::-1]
    gpu_dict = {}
    for i in range(100):
        idx = indices[i]
        word = char_array[hash_start[idx]:hash_start[idx]+hash_length[idx]]
        word = word.tobytes().decode('utf-8')
        gpu_dict[word] = counts[idx]
    print(gpu_dict)