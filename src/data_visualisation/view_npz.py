import numpy as np

# Replace 'your_file.npz' with the actual file name
file_name = 'Burgers.npz'

# Load the contents of the .npz file
data = np.load(file_name)

# Display the keys present in the .npz file
print("Keys in the .npz file:", data.files)

# Access and print the content associated with a specific key
# Replace 'key_name' with the actual key you want to access
key_name = 't'
if key_name in data:
    print(f"Content of '{key_name}':")
    print(data[key_name])
else:
    print(f"Key '{key_name}' not found in the .npz file.")

# Don't forget to close the loaded .npz file
data.close()