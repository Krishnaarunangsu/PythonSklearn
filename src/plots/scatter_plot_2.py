import numpy as np
from matplotlib import  pyplot as plt

x=np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y1=np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])
y2=np.array([95,87,89,86,112,84,101,90,95,79,80,83,82])

plt.scatter(x, y1)
plt.scatter(x, y2)
plt.xlabel('X')
plt.ylabel('Y')

plt.legend(['x*2', 'x*3'],loc="lower right")
plt.show()