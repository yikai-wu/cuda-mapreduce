import numpy as np
import sys

def generate_data(size_in_mb, num_features):
    # Each float32 element occupies 4 bytes
    bytes_per_element = 4
    # Total number of elements in the array
    total_elements = (size_in_mb * (1024**2)) // bytes_per_element
    # Number of rows is calculated by dividing total elements by the number of features
    num_rows = total_elements // num_features
    # Generate random float32 numbers
    data = np.random.rand(num_rows, num_features).astype(np.float32)
    # Verify the size
    data_size = data.nbytes  # nbytes gives the total bytes consumed by the data elements
    print(f"Generated data of size: {data_size / (1024**2)} MB")
    return data

if __name__ == "__main__":
    np.random.seed(args.random_seed)
    size_in_mb = 1000  # Specify the desired size in MB
    num_features = 2000
    data = generate_data(size_in_mb, num_features)
    # Use numpy.savetxt to write array to file in text format
    np.savetxt("num_data.txt", data, fmt='%f')
