import numpy as np
import os
ip = 0.476475
NumDomain = int(2000)
print(NumDomain // (1/ip))

a = np.array([1, 2, 1, 2, 1])
b = np.array([1, 2, 0, 4, 100])
# print((a+b)/2)
# print(np.maximum(a,b))
print(np.sqrt((a**2)+(b**2)))
quit()