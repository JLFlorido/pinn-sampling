import time
import numpy as np
import multiprocessing


def run_debugging():

    time_cost = []

    for n in range(5):
        start_t = time.time()
        time.sleep(0.05)
        time_cost.append(time.time() - start_t)
        # print(
        #     "--------------------------------------------\nThat was run {}".format(n)
        # )
        # print("Time taken: {:.02f}s".format(time.time() - start_t))

    print("\n{}".format(time_cost))
    # np.savetxt(f"results/raw/debugging/time_debugging", time_cost)


if __name__ == "__main__":
    for i in range(50):
        p = multiprocessing.Process(target=run_debugging)
        p.start()
        p.join()
        print(p.is_alive())

    # for i in range(50):
    #     run_debugging()
