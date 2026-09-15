# PyTorch简介

## 什么是PyTorch

PyTorch是一个开源的Python机器学习库，基于Torch库（一个有大量机器学习算法支持的科学计算框架，有着与Numpy类似的张量（Tensor）操作，采用的编程语言是Lua），底层由C++实现，应用于人工智能领域，如计算机视觉和自然语言处理。

PyTorch主要有两大特征：

- 类似于NumPy的张量计算，能在GPU或MPS等硬件加速器上加速。
- 基于带自动微分系统的深度神经网络。

PyTorch官网：https://pytorch.org/。

## PyTorch安装

> 【占位符】本节 PyTorch 安装相关内容（CPU 版本安装、GPU 版本安装、CUDA 版本选择与安装、PyTorch 安装步骤等）暂未提取，待后续补充。

## 张量创建

Tensor（张量）是PyTorch的核心数据结构。张量在不同学科中有不同的意义，在深度学习中张量表示一个多维数组，是标量、向量、矩阵的拓展。如一个RGB图像的数组就是一个三维张量，第1维是图像的高，第2维是图像的宽，第3维是图像的颜色通道。

### 基本张量创建

#### torch.tensor(data)创建指定内容的张量

```python
import torch
import numpy as np

# 创建标量张量
tensor1 = torch.tensor(10)
print(tensor1)

# 使用列表创建张量
tensor2 = torch.tensor([1, 2, 3])
print(tensor2)

# 使用numpy创建张量
tensor3 = torch.tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
print(tensor3)
```

#### torch.Tensor(size)创建指定形状的张量

```python
import torch

# 创建指定形状的张量，默认类型为float32
tensor1 = torch.Tensor(3, 2, 4)
print(tensor1)
print(tensor1.dtype)

# 也可以用来创建指定内容的张量
tensor2 = torch.Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
print(tensor2)
```

#### 创建指定类型的张量

可通过torch.IntTensor()、torch.FloatTensor()等创建。

或在torch.tensor()中通过dtype参数指定类型。

```python
import torch

# 创建int32类型的张量
tensor1 = torch.IntTensor(2, 3)
tensor2 = torch.tensor([1, 2, 3], dtype=torch.int32)
print(tensor1)
print(tensor2)

# 元素类型不匹配则会进行类型转换
tensor1 = torch.IntTensor([1.1, 2.2, 3.6])
tensor2 = torch.tensor([3.1, 2.2, 1.6], dtype=torch.int32)
print(tensor1)
print(tensor2)

# 创建int64类型的张量
tensor1 = torch.LongTensor([1, 2, 3])
tensor2 = torch.tensor([1, 2, 3], dtype=torch.int64)
print(tensor1, tensor1.dtype)
print(tensor2, tensor1.dtype)

# 创建int16类型的张量
tensor1 = torch.ShortTensor(2, 2)
tensor2 = torch.tensor([1, 2, 3], dtype=torch.int16)
print(tensor1, tensor1.dtype)
print(tensor2, tensor1.dtype)

# 创建float32类型的张量
tensor1 = torch.FloatTensor([9, 8, 7])
tensor2 = torch.tensor([1, 2, 3], dtype=torch.float32)
print(tensor1, tensor1.dtype)
print(tensor2, tensor1.dtype)

# 创建float64类型的张量
tensor1 = torch.DoubleTensor(2, 3, 1)
tensor2 = torch.tensor([1, 2, 3], dtype=torch.float64)
print(tensor1)
print(tensor2)
```

### 指定区间的张量创建

#### torch.arange(start, end, step)在区间内按步长创建张量

```python
import torch

# torch.arange(start, end, step) 在区间[start,end)中创建步长为step的张量
tensor1 = torch.arange(10, 30, 2)
print(tensor1)

# torch.arange(end) 创建区间为[0,end)，步长为1的张量
tensor2 = torch.arange(6)
print(tensor2)
```

#### torch.linspace(start, end, steps)在区间内按元素数量创建张量

```python
import torch

# torch.linspace(start, end, steps) 在区间按元素数量创建张量
tensor1 = torch.linspace(10, 30, 5)
print(tensor1)
```

#### torch.logspace(start, end, steps, base)在指数区间内按指定底数创建张量

```python
import torch

# torch.logspace(start, end, steps, base) 在区间[start,end]之间生成steps个数，并以base为底，区间内的数为指数创建张量
tensor1 = torch.logspace(1, 3, 3, 2)
print(tensor1)
```

### 按数值填充张量

- torch.zeros(size)创建指定形状的全0张量
- torch.ones(size)创建指定形状的全1张量
- torch.full(size, value)创建指定形状的按指定值填充的张量
- torch.empty(size)创建指定形状的未初始化的张量
- torch.zeros_like(input)创建与给定张量形状相同的全0张量
- torch.ones_like(input)创建与给定张量形状相同的全1张量
- torch.full_like(input, value)创建与给定张量形状相同的按指定值填充的张量
- torch.empty_like(input)创建与给定张量形状相同的未初始化的张量

```python
import torch

# torch.zeros(size) 创建指定形状的全0张量
tensor1 = torch.zeros(2, 3)
print(tensor1)

# torch.ones_like(input) 创建与给定张量形状相同的全1张量
tensor2 = torch.ones_like(tensor1)
print(tensor2)

# torch.full(size,fill_value) 创建指定形状的按指定值填充的张量
tensor1 = torch.full((2, 3), 6)
print(tensor1)

# torch.empty_like(input) 创建与给定张量形状相同的未初始化的张量
tensor2 = torch.empty_like(tensor3)
print(tensor2)
```

- torch.eye(n, [m])创建单位矩阵

```python
import torch

# torch.eye(n) 创建n*n的单位矩阵
tensor1 = torch.eye(3)
print(tensor1)

# torch.eye(n, m) 按指定的行和列创建
tensor2 = torch.eye(3, 4)
print(tensor2)
```

### 随机张量创建

- torch.rand(size)创建在[0,1)上均匀分布的，指定形状的张量
- torch.randint(low, high, size)创建在[low,high)上均匀分布的，指定形状的张量
- torch.randn(size)创建标准正态分布的，指定形状的张量
- torch.normal(mean,std,size)创建自定义正态分布的，指定形状的张量
- torch.rand_like(input)创建在[0,1)上均匀分布的，与给定张量形状相同的张量
- torch.randint_like(input, low, high)创建在[low,high)上均匀分布的，与给定张量形状相同的张量
- torch.randn_like(input)创建标准正态分布的，与给定张量形状相同的张量

```python
import torch

# torch.rand(size) 创建在[0,1)上均匀分布的，指定形状的张量
tensor1 = torch.rand(2, 3)
print(tensor1)

# torch.rand_like(input) 创建在[0,1)上均匀分布的，与给定张量形状相同的张量
tensor2 = torch.randint_like(tensor1, 1, 10)
print(tensor2)

# torch.randn(size) 创建标准正态分布的，指定形状的张量
tensor1 = torch.randn(4, 2)
print(tensor1)

# torch.normal(mean,std,size) 创建自定义正态分布的，指定形状的张量。mean为均值，std为标准差
tensor2 = torch.normal(5, 1, tensor1.shape)
print(tensor2)
```

- torch.randperm(n)生成从0到n-1的随机排列，类似洗牌

```python
import torch

# torch.randperm(n) 生成从0到n-1的随机排列
tensor1 = torch.randperm(10)
print(tensor1)
```

- torch.random.initial_seed()查看随机数种子
- torch.manual_seed(seed)设置随机数种子

```python
import torch

# 查看随机数种子
print(torch.random.initial_seed())
# 设置随机数种子
torch.manual_seed(42)
print(torch.random.initial_seed())
```

## 张量转换

### 张量元素类型转换

#### Tensor.type(dtype)修改张量的类型

```python
import torch

tensor1 = torch.tensor([1, 2, 3])
print(tensor1, tensor1.dtype)

# 使用type方法修改张量的类型
tensor1 = tensor1.type(torch.float32)
print(tensor1, tensor1.dtype)
```

#### Tensor.double()等修改张量的类型

```python
import torch

tensor1 = torch.tensor([1, 2, 3])
print(tensor1, tensor1.dtype)

# 使用double方法修改张量的类型
tensor1 = tensor1.double()
print(tensor1)
# 使用long方法修改张量的类型
tensor1 = tensor1.long()
print(tensor1, tensor1.dtype)
```

### Tensor与ndarray转换

#### Tensor.numpy()将Tensor转换为ndarray，共享内存。使用copy()避免共享内存

```python
import torch

# 使用numpy()方法将Tensor转换为ndarray，共享内存
tensor1 = torch.rand(3, 2)
numpy_array = tensor1.numpy()
print(tensor1)
print(numpy_array)
print(type(tensor1), type(numpy_array))
print()
tensor1[:, 0] = 4
print(tensor1)
print(numpy_array)
print()

# 使用copy()方法避免共享内存
numpy_array = tensor1.numpy().copy()
tensor1[:, 0] = -1
print(tensor1)
print(numpy_array)
```

#### torch.from_numpy(ndarray)将ndarray转换为Tensor，共享内存。使用copy()避免共享内存

```python
import torch
import numpy as np

# 使用from_numpy()方法将ndarray转换为Tensor，共享内存
numpy_array = np.random.randn(3)
tensor1 = torch.from_numpy(numpy_array)
print(numpy_array)
print(tensor1)
print()
numpy_array[0] = 100
print(numpy_array)
print(tensor1)
print()

# 使用copy()方法避免共享内存
tensor1 = torch.from_numpy(numpy_array.copy())
numpy_array[0] = -1
print(numpy_array)
print(tensor1)
```

#### torch.tensor(ndarray)将ndarray转换为Tensor，不共享内存

```python
import torch
import numpy as np

# 使用torch.tensor()将ndarray转换为Tensor
numpy_array = np.random.randn(3)
tensor1 = torch.tensor(numpy_array)
print(numpy_array)
print(tensor1)
print()
numpy_array[0] = 100
print(numpy_array)
print(tensor1)
```

### Tensor与标量转换

若张量中只有1个元素，Tensor.item()可提取张量中元素为标量。

```python
import torch

tensor1 = torch.tensor(1)
print(tensor1)
print(tensor1.item())
```

## 张量数值计算

### 基本运算

#### 四则运算

- +、-、*、/加减乘除
- add()、sub()、mul()、div()加减乘除，不改变原数据
- add_()、sub_()、mul_()、div_()加减乘除、修改原数据

```python
import torch

tensor1 = torch.randint(1, 9, (2, 3))
print(tensor1)
print(tensor1 + 10)
print()

# add()，不修改原数据
print(tensor1.add(10))
print(tensor1)
print()

# add_()，修改原数据
print(tensor1.add_(10))
print(tensor1)
```

#### -、neg()、neg_()取负

```python
import torch

tensor1 = torch.tensor([1, 2, 3])
print(-tensor1)
print()

print(tensor1.neg())
print(tensor1)
print()

print(tensor1.neg_())
print(tensor1)
```

#### **、pow()、pow_()求幂

```python
import torch

tensor1 = torch.tensor([1, 2, 3])
print(tensor1**2)
print()

print(tensor1.pow(2))
print(tensor1)
print()

print(tensor1.pow_(2))
print(tensor1)
```

#### sqrt()、sqrt_()求平方根

```python
import torch

tensor1 = torch.tensor([1.0, 2.0, 3.0])
print(tensor1.sqrt())
print(tensor1)
print()

print(tensor1.sqrt_())
print(tensor1)
```

#### exp()、exp_()以e为底数求幂

```python
import torch

tensor1 = torch.tensor([1.0, 2.0, 3.0])
print(2.71828183**tensor1)
print()

print(tensor1.exp())
print(tensor1)
print()

print(tensor1.exp_())
print(tensor1)
```

#### log()、log_()以e为底求对数

```python
import torch

tensor1 = torch.tensor([1.0, 2.0, 3.0])
print(tensor1.log())
print(tensor1)
print()

print(tensor1.log_())
print(tensor1)
```

### 哈达玛积（元素级乘法）

两个矩阵对应位置元素相乘称为哈达玛积（Hadamard product）。

使用*、mul()实现两个形状相同的张量之间对位相乘。

```python
import torch

tensor1 = torch.tensor([[1, 2], [3, 4]])
tensor2 = torch.tensor([[1, 2], [3, 4]])
print(tensor1 * tensor2)
print(tensor1.mul(tensor2))
```

### 矩阵乘法运算

mm()严格用于二维矩阵相乘。

@、matmul()支持多维张量，按最后两个维度做矩阵乘法，其他维度相同，或者至少一个张量对应维度为1，广播后进行运算。

```python
import torch

# 2维矩阵的矩阵乘法
tensor1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
tensor2 = torch.tensor([[1, 2], [3, 4], [5, 6]])
print(tensor1)
print(tensor2)
print(tensor1.mm(tensor2))
print(tensor1 @ tensor2)
print(tensor1.matmul(tensor2))
print()

# 3维张量的矩阵乘法
tensor1 = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[6, 5, 4], [3, 2, 1]]])
tensor2 = torch.tensor([[[1, 2], [3, 4], [5, 6]], [[6, 5], [4, 3], [2, 1]]])
print(tensor1)
print(tensor2)
print(tensor1 @ tensor2)
print(tensor1.matmul(tensor2))
```

## 张量运算函数

常见运算函数：

sum()求和

mean()求均值

max()/min()求最大/最小值及其索引

argmax()/argmin()求最大值/最小值的索引

std()求标准差

unique()去重

sort()排序

```python
import torch

tensor1 = torch.randint(1, 9, (3, 2, 4))
tensor1 = tensor1.float()
print(tensor1)
print()

# sum() 求和
print("求和")
print(tensor1.sum())
print("按第0个维度求和")
print(tensor1.sum(dim=0))
print()

# mean() 求均值)
print("求均值")
print(tensor1.mean())
print("按第1个维度求均值")
print(tensor1.mean(dim=1))
print()

# max() 求最大值
print("求最大值")
print(tensor1.max())
print("按第2个维度求最大值与索引")
print(tensor1.max(dim=2))
print()

# argmin() 求最小值索引
print("求最小值索引")
print(tensor1.argmin())
print()

# std() 求标准差
print("求标准差")
print(tensor1.std())
print()

# unique() 去重
print("去重")
print(tensor1.unique())
print()

# sort() 排序
print("排序")
print(tensor1.sort())
```

## 张量索引操作

### 简单索引

```python
import torch

tensor1 = torch.randint(1, 9, (3, 5, 4))
print(tensor1)
print()

# 取 第0维第0
print(tensor1[0])
print()

# 取 第0维所有，第1维第1
print(tensor1[:, 1])
print()

# 取 第0维所有，第1维第1，第2维第3
print(tensor1[2, 1, 3])
```

### 范围索引

```python
import torch

tensor1 = torch.randint(1, 9, (3, 5, 4))
print(tensor1)
print()

# 取 第0维第1到最后
print(tensor1[1:])
print()

# 取 第0维最后，第1维1到3(包含3),第2维0到2(包含2)
print(tensor1[-1:, 1:4, 0:3])
print()
```

### 列表索引

```python
import torch

tensor1 = torch.randint(1, 9, (3, 5, 4))
print(tensor1)
print()

# 取 第0维第0，第1维第1 和 第0维第1，第1维第2
print(tensor1[[0, 1], [1, 2]])
print()

# 取 第0维第0，第1维第1、2 和 第0维第1，第1维第1、2
print(tensor1[[[0], [1]], [1, 2]])
```

### 布尔索引

```python
import torch

tensor1 = torch.randint(1, 9, (3, 5, 4))
print(tensor1)
print()

# 取 第2维第0大于5的，返回(dim0,dim1)形状的索引
print(tensor1[:, :, 0] > 5)
print(tensor1[tensor1[:, :, 0] > 5])
print()

# 取 第1维第1大于5的，返回(dim0,dim2)形状的索引
mask = tensor1[:, 1, :] > 5
print(mask)
tensor2 = tensor1.permute(0, 2, 1)  # 转换维度为(dim0,dim2,dim1)
print(tensor2[mask])
tensor2 = tensor2[mask].permute(1, 0)  # 转换维度为(dim1,?)
print(tensor2)
print()

# 取 第1维第1，第2维第2大于5的，返回(dim0)形状的索引
print(tensor1[:, 1, 2] > 5)
print(tensor1[tensor1[:, 1, 2] > 5])
```

## 张量形状操作

### 交换维度

#### transpose()交换两个维度

```python
import torch

tensor1 = torch.randint(1, 9, (2, 3, 6))
print(tensor1)
print(tensor1.transpose(1, 2))  # 交换第1维和第2维
```

#### permute()重新排列多个维度

```python
import torch

tensor1 = torch.randint(1, 9, (2, 3, 6))
print(tensor1)
print(tensor1.permute(2, 0, 1))  # (2, 3, 6)->(6, 2, 3)
```

### 调整形状

#### reshape()调整张量的形状

```python
import torch

tensor1 = torch.randint(1, 9, (3, 5, 4))
print(tensor1)
print(tensor1.reshape(6, 10))
print(tensor1.reshape(3, -1))
```

#### view()调整张量的形状，需要内存连续。共享内存

is_contiguous()判断是否内存连续

contiguous()转换为内存连续

```python
import torch

tensor1 = torch.randint(1, 9, (3, 5, 4))
print(tensor1)
print(tensor1.is_contiguous())  # is_contiguous()判断是否内存连续
print(tensor1.view(-1, 10))

tensor1 = tensor1.T
print(tensor1.is_contiguous())  # is_contiguous()判断是否内存连续
print(tensor1.contiguous().view(-1))  # contiguous()强制内存连续
```

### 增加或删除维度

#### unsqueeze()在指定维度上增加1个维度

```python
import torch

tensor1 = torch.tensor([1, 2, 3, 4, 5])
print(tensor1)
# 在0维上增加一个维度
print(tensor1.unsqueeze(dim=0))
# 在1维上增加一个维度
print(tensor1.unsqueeze(dim=1))
# 在-1维上增加一个维度
print(tensor1.unsqueeze(dim=-1))
```

#### squeeze()删除大小为1的维度

```python
import torch

tensor1 = torch.tensor([1, 2, 3, 4, 5])
print(tensor1.unsqueeze_(dim=0))
print(tensor1.squeeze())
```

## 张量拼接操作

#### torch.cat()张量拼接，按已有维度拼接。除拼接维度外，其他维度大小须相同

```python
import torch

tensor1 = torch.randint(1, 9, (2, 2, 5))
tensor2 = torch.randint(1, 9, (2, 1, 5))
print(tensor1)
print(tensor2)
print(torch.cat([tensor1, tensor2], dim=1))
```

#### torch.stack()张量堆叠，按新维度堆叠。所有张量形状必须一致

```python
import torch

torch.manual_seed(42)
tensor1 = torch.randint(1, 9, (3, 1, 5))
tensor2 = torch.randint(1, 9, (3, 1, 5))
print(tensor1)
print(tensor2)
tensor3 = torch.stack([tensor1, tensor2], dim=2)
print(tensor3)
print(tensor3.shape)
```
