import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from mpl_toolkits.mplot3d import Axes3D
"""
x = np.arange(10)
y = np.random.randn(10)
plt.scatter(x, y, color='red', marker='+')
plt.savefig('plot/'+'collisio2n.jpg')

x = np.arange(100)
y = np.random.randn(100)
plt.scatter(x, y, color='red', marker='+')
plt.savefig('plot/'+'collision.jpg')
"""
A = np.arange(12).reshape((2, 6))
A = np.split(A, 3, axis=1)
dates = pd.date_range('20160101', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A', 'B', 'C', 'D'])
df.iloc[0, 1] = np.nan
# data = pd.Series(np.random.randn(1000), index=np.arange(1000))
data = pd.DataFrame(
    np.random.randn(1000, 4),
    index=np.arange(1000),
    columns=list("ABCD")
)
data.cumsum()
ar = np.array([[1,2],[3,5]])
print(ar*3)
