import time
import numpy as np

time_cost = []
for n in range(10):
    start_t = time.time()
    time.sleep(1.5)
    time_cost.append(time.time() - start_t)
    print("\n --------------------------------------------\nThat was run {}".format(n))
    print("Time taken: {:.02f}s".format(time.time() - start_t))

print("\n{}".format(time_cost))
np.savetxt(f"results/raw/debugging/time_debugging", time_cost)
