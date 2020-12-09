import numpy as np
import math

data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
size = int(math.sqrt(len(data)))
A = np.array(data)
B = np.reshape(A, (size, size))

for i in range(size - 1):
    x = [k for k in range(0, size - i)]
    y = [k for k in range(i, size)]
    print(B[x, y])
    print(B[x, [(size - k - 1) for k in y]])
    if i > 0:
        print(B[y, x])
        print(B[[(size - k - 1) for k in x], y])
