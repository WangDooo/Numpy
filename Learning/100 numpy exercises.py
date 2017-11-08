# 1.Print the numpy version and the configuration
import numpy as np 

# 2.Print the numpy version and the configuration
# print(np.__version__)
# np.show_config()

# 3.Create a null vector of size 10
# Z = np.zeros(10)
# print(Z)

# 4.How to get the documentation of the numpy add function from the command line? 
# python -c "import numpy; numpy.info(numpy.add)"

# 5.Create a null vector of size 10 but the fifth value which is 1
# Z = np.zeros(10)
# Z[4] = 1
# print(Z)
 
# 6.Create a vector with values ranging from 10 to 49
# Z = np.arange(10,50)
# print(Z)

# 7.Reverse a vector (first element becomes last)
# Z = np.arange(50)
# Z = Z[::-1]
# print(Z)

# 8.Create a 4x3 matrix with values ranging from 0 to 11
# Z = np.arange(12).reshape(4,3)
# print(Z)

# 9.Find indices of non-zero elements from [1,2,0,0,4,0] 
# nz = np.nonzero([1,2,0,0,4,0])
# print(nz)

# 10.Create a 3x3 identity matrix
# Z = np.eye(3)
# print(Z)

# 11.Create a 3x3x3 array with random values
# Z = np.random.random((3,3,3))
# print(Z)

# 12.Create a 10x10 array with random values and find the minimum and maximum values
# Z = np.random.random((10,10))
# Zmin, Zmax = Z.min(), Z.max()
# print(Zmin, Zmax)

# 13.Create a random vector of size 30 and find the mean value 
# Z = np.random.random(30)
# m = Z.mean()
# print(Z, m)

# 14.Create a 2d array with 1 on the border and 0 inside 
# Z = np.ones((10,10))
# Z[1:-1, 1:-1] = 0
# print(Z)

# 15.What is the result of the following expression?
# print(0 * np.nan) 
# print(np.nan == np.nan)
# print(np.inf > np.nan)
# print(np.nan - np.nan)
# print(0.3 == 3 * 0.1)
# print(np.isnan(np.nan))

# 16.Create a 5x5 matrix with values 1,2,3,4 just below the diagonal
# Z = np.diag(1+np.arange(4),k=-1) # k=0 on the diagonal
# print(Z)

# 17.Create a 8x8 matrix and fill it with a checkerboard pattern
# Z = np.zeros((8,8), dtype=int)
# Z[1::2,::2] = 1
# Z[::2,1::2] = 1
# print(Z)

# 18.Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element
# print(np.unravel_index(100,(6,7,8))) # fill begin with 0

# 19.Create a checkerboard 8x8 matrix using the tile function t
# Z = np.tile(np.array([[0,1],[1,0]]), (4,4)) # tile 分块重复
# print(Z)

# 20.Normalize a 5x5 random matrix
# Z = np.random.random((5,5))
# Zmax, Zmin = Z.max(), Z.min()
# Z = (Z - Zmin)/(Zmax - Zmin)
# print(Z)

# 21.Create a custom dtype that describes a color as four unisgned bytes (RGBA)
# color = np.dtype([("r", np.ubyte, 1),
# 				  ("g", np.ubyte, 1),
# 				  ("b", np.ubyte, 1),
# 				  ("a", np.ubyte, 1)])
# print(color)

# 22.Multiply a 5x3 matrix by a 3x2 matrix (real matrix product)
# Z = np.dot(np.ones((5,3)), np.ones((3,2)))
# print(Z)

# 23.Given a 1D array, negate all elements which are between 3 and 8, in place
# Z = np.arange(11)
# Z[(3 < Z) & (Z < 8)] *= -1
# print(Z)

# 24.What is the output of the following script?
# print(sum(range(5),-1)) # 参数'start'的值（默认为0）此处是-1
# from numpy import *
# print(sum(range(5),-1)) # .sum()函数是模块numpy的一个函数：



# 32.Create a random vector of size 10 and sort it
# Z = np.random.random(10)
# Z.sort()
# print(Z)

