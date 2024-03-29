import numpy as np
from matplotlib import  pyplot as plt

x=np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y=np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])

plt.scatter(x, y)
plt.xlabel('X')
plt.ylabel('Y')

plt.legend(['x*2'],loc="lower right")
plt.show()