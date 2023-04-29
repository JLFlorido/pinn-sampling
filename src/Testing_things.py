import time
import numpy as np
import multiprocessing


def run_debugging():

    time_cost = []
    
    error = [0.1]
    for n in range(5):
        local_start_t = time.time()
        time.sleep(0.05)
        time_taken=time.time()-local_start_t
        error.append(time_taken)
        time_cost.append(time_taken)
        # print(
        #     "--------------------------------------------\nThat was run {}".format(n)
        # )
        # print("Time taken: {:.02f}s".format(time.time() - start_t))
        
    print("\nRound results:\n{}".format(time_cost))
    time_cost = np.array(time_cost)
    error=np.array(error)
    return error, time_taken
    # np.savetxt(f"results/raw/debugging/time_debugging", time_cost)
    

if __name__ == "__main__":
    time_cost=[]
    error = []
    for i in range(3):
        start_t = time.time()
        error1,_ = run_debugging()
        error.append(error1)
        time_cost.append((time.time() - start_t))
        print("Finished run #{}".format(i))
        print("Time taken: {:.04f}s\n--------------------------------------------\n".format(time.time() - start_t))
    error = np.array(error)
    print(error)
    print(error.shape)
    
    np.savetxt(f"results/raw/ignore_thisisatest.txt", error)