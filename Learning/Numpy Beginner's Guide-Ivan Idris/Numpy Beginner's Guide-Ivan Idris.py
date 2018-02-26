import numpy as np
import sys
from datetime import datetime


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

# <-----------股票收益率分析----------------------------------------->

# c = np.loadtxt('data.csv', delimiter=',', usecols=(6,), unpack=True)
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

# returns = np.diff(c) / c[ :-1] # diff() 返回一个由相邻数组元素的差值构成的数组 比原数组会少一个元素
# print("returns_diff(c) =", returns)
# print("Standard deviation =", np.std(returns)) # std() 计算标准差

# logreturns = np.diff(np.log(c)) # log() 取对数 一般情况要检查数组中 不含有零和负数

# posretindices = np.where(returns > 0) # where() 根据指定条件返回所有满足条件的数组元素索引值
# print("Indices with positive returns", posretindices)

# 波动率 volatility 历史波动率根据历史价格计算得到，历史波动率需要用到对数收益率。
# 年波动率 = 对数收益率的标准差除以其均值，再除以交易日倒数的平方根，通常交易日取252天

# annul_volatility = np.std(logreturns) / np.mean(logreturns)
# annul_volatility = annul_volatility / np.sqrt(1./252.) # 数字后面有个'.'
# print("Annual volatility", annul_volatility)

# print("Monthly volatility", annul_volatility * np.sqrt(1./12.))

# <-----------日期分析----------------------------------------->

<<<<<<< HEAD
# def datestr2num(s):
# 	return datetime.strptime(s.decode('ascii'), "%d-%m-%Y").date().weekday()
# 	# 编译器在打开data.csv文件时，将表格里的第2列数组值提取出来返回给dates，第二列值是日期格式字符串，但因为我们是以二进制编码的格式打开第二列值是，返回的值字节字符串bytes，所以需要把它便会string，则对字符串解码用函数decode('asii')，变成string格式。
=======
def datestr2num(s):
	return datetime.strptime(s.decode('ascii'), "%d-%m-%Y").date().weekday()
	# 编译器在打开data.csv文件时，将表格里的第2列数组值提取出来返回给dates，第二列值是日期格式字符串，但因为我们是以二进制编码的格式打开第二列值是，返回的值字节字符串bytes，所以需要把它便会string，则对字符串解码用函数decode('asii')，变成string格式。
>>>>>>> d0a685bb86a26f1cf8f1c9ad515b16822be2cc30

# dates, close = np.loadtxt('data.csv', delimiter=',', usecols=(1,6), converters={1:datestr2num}, unpack=True)
# print("Dates =", dates) # converters参数 数据列和转换函数之间进行映射

# averages = np.zeros(5)
# for i in range(5):
# 	indices = np.where(dates == i)
# 	prices = np.take(close, indices) # take() 按照where的索引值获取对应的元素
# 	avg = np.mean(prices)
# 	print("Day", i, "prices", prices, "Average", avg)
# 	averages[i] = avg

# top = np.max(averages)
# print("Highest average =", top, "Top day of the week is", np.argmax(averages)) # argmax() 返回数组中最大元素的索引值
# bottom = np.min(averages)
# print("Lowest average =", bottom, "Bottom day of the week is", np.argmin(averages))

# <-----------周汇总---------------------------------------->

# dates, open, high, low, close = np.loadtxt('data.csv', delimiter=',', usecols=(1,3,4,5,6), converters={1:datestr2num}, unpack=True)

# close = close[:16]
# dates = dates[:16]

# first_monday = np.ravel(np.where(dates == 0))[0] # where() 返回的是一个多维数组 用ravel函数将其展平
# print("The first Monday index is", first_monday)

# last_friday = np.ravel(np.where(dates == 4))[-1]
# print("The last Firday index is", last_friday)

# weeks_indices = np.arange(first_monday, last_friday+1)
# print("Week indices initial", weeks_indices)

# weeks_indices = np.split(weeks_indices, 3) # split() 分割数组 分成3个组
# print("Weeks indices after split", weeks_indices)

# def summarize(a, o, h, l, c):
# 	monday_open = o[a[0]]
# 	week_high = np.max(np.take(h, a))
# 	week_low = np.min(np.take(l, a))
# 	friday_close = c[a[-1]]

# 	return("APPL", monday_open, week_high, week_low, friday_close)

# weeksummary = np.apply_along_axis(summarize, 1, weeks_indices, open, high, low, close)
# # apply_along_axis() 提供我们自定义的函数名summarize, 指定要作用的轴或维度的编号, summarize函数的参数(第一个为目标数组)
# print("Week summary", weeksummary)

# np.savetxt("weeksummary.csv", weeksummary, delimiter=',', fmt="%s") # savetxt() 存储 指定了 文件名\保存数组名\分隔符\存储格式
# # 存储格式 fmt="%?" ?: -左对齐 0左端补0 +输出符号 c单个字符 d十进制有符号整数 u十进制无符号 s字符串 f浮点数 e科学计数法 o八进制有符号整数 x十六进制无符号整数

# <-----------真实波动浮动均值ATR---------------------------------------->

# h, l, c = np.loadtxt('data.csv', delimiter=',', usecols=(4,5,6), unpack=True)

# N = 20
# h = h[-N:]
# l = l[-N:]

# print("len(h) =", len(h), "len(l) =", len(l))

# previousclose = c[-N-1: -1]
# print("len(previousclose)", len(previousclose))
# print("Previous close", previousclose)

# truerange = np.maximum(h - l, h - previousclose, previousclose - l) # maximum() 在几个数组之间，挑选每个元素位置上的最大值
# print("True range", truerange)

# atr = np.zeros(N)
# atr[0] = np.mean(truerange)
# for i in range(1, N):
# 	atr[i] = (N - 1) * atr[i -1] + truerange[i]
# 	atr[i] /= N

# print("ATR", atr)

# <-----------简单移动平均线---------------------------------------->

from matplotlib.pyplot import plot
from matplotlib.pyplot import show

# N = 5

# weights = np.ones(N) / N
# print("Weights", weights)
# c = np.loadtxt('data.csv', delimiter=',', usecols=(6,), unpack=True)

# sma = np.convolve(weights, c)[N-1: -N+1] # convolve() 计算一数组与指定权重的卷积
# t = np.arange(N - 1, len(c))
# plot(t, c[N-1:], lw=1.0)
# plot(t, sma, lw=2.0)
# show()

# <-----------指数移动平均线---------------------------------------->

x = np.arange(5)
print("Exp", np.exp(x))
print("Linspace", np.linspace(-1, 0, 5)) # linspace() 起始值，终止值，元数个数， 返回一个元素值在指定范围内均匀分布的数组

N = 5

weights = np.exp(np.linspace(-1, 0, N))
weights /= weights.sum() # 对权重进行归一化，用ndarray对象的sum()方法
print("Weights", weights)

c = np.loadtxt('data.csv', delimiter=',', usecols=(6,), unpack=True)
ema = np.convolve(weights, c)[N-1: -N+1] # convolve() 计算一数组与指定权重的卷积
t = np.arange(N - 1, len(c))
plot(t, c[N-1:], lw=1.0)
plot(t, ema, lw=2.0)
show()