import numpy as np
import os

error_hist = np.array([((0.5+np.random.rand())/1000),((0.2+np.random.rand())/1000),(np.random.rand()/1000)])
final_error = (np.random.rand()/1000)
time_taken = (1000*np.random.rand())+15500

if np.isscalar(time_taken):
    time_taken = np.atleast_1d(time_taken)
if np.isscalar(final_error):
    final_error = np.atleast_1d(final_error)

# Define output directory and file names. Should come from doc opts further on.
print("Current working directory:", os.getcwd())
output_dir = "./testing_folder"  # Replace with your desired output directory path
error_hist_fname = "error_hist_test.txt"
error_final_fname = "error_final_test.txt"
time_taken_fname = "time_taken_test.txt"

# If results directory does not exist, create it
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Define the full file paths
error_hist_fname = os.path.join(output_dir, error_hist_fname)
error_final_fname = os.path.join(output_dir, error_final_fname)
time_taken_fname = os.path.join(output_dir, time_taken_fname)

# Define function to append to file
def append_to_file(file_path, data):
    with open(file_path, 'ab') as file:
        file.write(b"\n")
        np.savetxt(file,data, newline=", ")
        # file.write(np.array2string(data, separator='\t') + '\n')

# Use function to append to file.
append_to_file(error_hist_fname, error_hist)
append_to_file(error_final_fname, final_error)
append_to_file(time_taken_fname, time_taken)