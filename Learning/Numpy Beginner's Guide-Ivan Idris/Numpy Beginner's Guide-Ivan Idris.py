import numpy as np

# m = np.array([np.arange(2),np.arange(2)])
# print(m)

# print(m.dtype.itemsize)

# 自定义数据类型
# t = np.dtype([('name', np.str_, 40), ('numitems', np.int32), ('price', np.float32)])
# print(t)

# items = np.array([('Wang', 3, 4.3), ('Doo', 11, 6.5)], dtype=t)
# print(items)

# <-----一维数组的索引和切片------------------------------------------------------->
# a = np.arange(9)
# print(a[3:7])
# print(a[:7:2])
# print(a[::-1]) # the same as python 用负数下标翻转数组

# <-----多为数组的切片和索引------------------------------------------------------->
# b = np.arange(24).reshape(2,3,4)
# print(b)
# print(b[0,0,1])
# print(b[:,0,0])
# print(b[0,:,:])
# print(b[0, ...]) # 多个冒号可以用一个省略号(...)代替
# print(b[0,1]) # [4 5 6 7] 第一层第二排

# print(b[0,1,::2])
# print(b[...,1])
# print(b[:,1])
# print(b[0,:,1])
# print(b[0,:,-1])
# print(b[0,::-1,-1])
# print(b[0,::2,-1])
# print(b[::-1])


# <-----改变数组维度------------------------------------------------------->
# b.ravel() # ravel 展平数组 只返回数组的一个视图，不改变b
# print(b)
# print(b.ravel())

# b.flatten() # flatten 展平 会请求分配内存来保存结果
# print(b)
# print(b.flatten())

# b.shape = (6,4)
# print(b)

# b.transpose() # 不改变b
# print(b)
# print(b.transpose())

# b.resize((2,12)) # resize 和 reshape功能一样，但resize会直接修改所操作的数组
# print(b)


# <---------组合数组----------------------------------------------->
# a = np.arange(9).reshape(3,3)
# print(a)
# b = a * 2
# print(b)

# c = np.hstack((a, b)) # hstack 水平组合
# print(c)
# d = np.concatenate((a, b ), axis = 1) # concatenate(axis = 1) 水平组合
# print(d)

# c = np.vstack((a, b)) # vstack 垂直组合
# print(c)
# d = np.concatenate((a, b), axis = 0) # concatenate(axis = 0) 垂直组合
# print(d)

# c = np.dstack((a, b))  # dstack 深度组合 例：若干二维数据，沿纵向叠在一起
# print(c)

# oned = np.arange(3)
# twice_oned = oned * 2
# c = np.column_stack((oned, twice_oned)) # column_stack 一维按列方向组合->两列
# print(c)

# print(np.column_stack((a, b)) == np.hstack((a, b))) # 二维和hstack相同 可以用"=="比较数组

# d = np.row_stack((oned, twice_oned)) # column_stack 一维按行方向组合->两行
# print(d)

# print(np.row_stack((a, b)) == np.vstack((a, b))) 


# <---------分隔数组----------------------------------------------->

# a = np.arange(9).reshape(3,3)
# b = np.hsplit(a, 3) # hsplit 把数组沿水平方向分割为3个相同大小的子数组
# print(b)
# c = np.split(a, 3, axis = 1)
# print(c)

# b = np.vsplit(a ,3)	# vsplit 把数组沿垂直方向分割为3个相同大小的子数组
# print(b)
# c = np.split(a, 3, axis = 0)
# print(c)

# <---------数组的属性----------------------------------------------->

# b = np.arange(24).reshape(2,12)
# print(b)

# print(b.ndim) # ndim 数组维数

# print(b.size) # size 数组元素总个数

# print(b.itemsize) # itemsize 数组中元素在内存中所占的字节数

# print(b.nbytes) # nbytes 整个数组所占的存储空间

# print(b.size * b.itemsize)

# b.resize(6,4)
# print(b)

# print(b.T) # T = transpose 转置		对于一维数组，T就是原数组

# b = np.array([1.j + 1, 2.j + 3]) # 复数的虚部用j表示
# print(b)

# print(b.real) # real 给出复数数组的实部

# print(b.imag) # imag 给出复数数组的虚部

# print(b.dtype) # 如果数组中包含复数元素，其数据类型自动变为复数型
# print(b.dtype.str)

# b = np.arange(4).reshape(2,2)
# print(b)
# f = b.flat # flat 返回一个numpy.flatiter对象
# print(f)
# for item in f:
# 	print(item)
# print(b.flat[2]) # 用flatiter对象直接获取一个数组元素

# b.flat = 7 # flat属性是一个可赋值的属性，赋值将会导致整个数组的元素都被覆盖
# print(b)

# b.flat[[1,3]] = 1 # 赋值多个元素 1 3 位置
# print(b)

# <---------数组的转换----------------------------------------------->

# b = np.array([1.j + 1, 2.j + 3])
# print(b.tolist()) # tolist() 将Numpy数组转换为Python列表

# print(b.astype(int)) # astype() 转换数组时指定数据类型 int会丢失去虚数部分，会有警告信息
# print(b.astype('complex')) 


# <--------------------------------------------------------------------->
# <-----------第3章- 常用函数------------------------------------------->
# <--------------------------------------------------------------------->

# <-----------读写文件txt------------------------------------------->

# i2 = np.eye(2)
# print(i2)
# np.savetxt("eye.txt", i2) # savetxt()

# <-----------读入CSV文件------------------------------------------->

# loadtxt() 读取csv文件 自动切分字段 并将数据载入Numpy数组
# c, v = np.loadtxt('data.csv', delimiter=',', usecols=(6,7) , unpack=True) 
# delimiter 分隔符 
# usecols	参数是一个元组，获取第7字段至第8字段数据 即收盘价和交易量
# unpack = True 分拆存储不同列的数据，即将收盘价和交易量赋值给c和v

# VWAP Volume-Weighted Average Price 成交量加权平均价格 以成交量为权重算出来的加权平均值

# vwap = np.average(c, weights=v) # 将v作为权重参数使用
# print("VWAP =", vwap)

# print("mean =", np.mean(c)) # mean() 算术平均值

# TWAP Time-Weighted Average Price 时间加权平均价格 基本思想：最近的价格重要性大一些，近期价格有较高权重

# t = np.arange(len(c))
# print("TWAP =", np.average(c, weights=t))

# h, l = np.loadtxt('data.csv', delimiter=',', usecols=(4,5), unpack=True)
# print("Highest =", np.max(h))
# print("Lowest =", np.min(l))
# print("区间中点 = ", (np.max(h) + np.min(l)) / 2)

# print("Spread high price =", np.ptp(h)) # ptp() 计算数组的取值范围 = max(array)-min(array)
# print("Spread Low price =", np.ptp(l))


c = np.loadtxt('data.csv', delimiter=',', usecols=(6,), unpack=True)
# print("Median =", np.median(c)) # median() 中位数

# sorted_c = np.msort(c)
# print("sorted =", sorted_c)
# N = len(c)
# if N % 2 == 0:
# 	print("average middle =",(sorted_c[int(N/2)] + sorted_c[int(N/2) - 1]) /2)
# else:
# 	print("average middle =",(sorted_c[int((N-1))/2]))

# print("variance =", np.var(c)) # var() 方差 各个数据与所有数据算术平均数的差的平方和除以数据个数

# print("variance from definition =", np.mean((c - c.mean())**2)) # c.mean() ndarray对象有mean方法

returns = np.diff(c) / c[ :-1] # diff() 返回一个由相邻数组元素的差值构成的数组 比原数组会少一个元素
print("Standard deviation =", np.std(returns)) # std() 计算标准差