import numpy as np
import os
import time

error_final_fname = "testing_folder/test.txt"
time_taken_fname = "testing_folder/test2.txt"
error_final = np.random.rand()*3.1
time_taken = np.random.rand()*100+1243.1
if np.isscalar(error_final):
    error_final = np.atleast_1d(error_final)
if np.isscalar(time_taken):    
    time_taken = np.atleast_1d(time_taken)

def append_to_file(file_path, data):
    try:    
        with open(file_path, 'ab') as file:
            np.savetxt(file, data)
    except Exception as e:
        print(f"An exception occurred: {e}")
        print("Retrying in 5 seconds...")
        time.sleep(5)
        try:
            with open(file_path, 'ab') as file:
                np.savetxt(file, data, newline=", ")
                file.write(b"\n")
        except Exception as e2:
            print(f"An exception occurred again: {e2}")

# Use function to append to file.
append_to_file(error_final_fname, error_final)
append_to_file(time_taken_fname, time_taken)