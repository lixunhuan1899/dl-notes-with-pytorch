import os
import math
import numpy as np
import time

import matplotlib.pyplot as plt

#Ipython为notebook环境而设置
#from IPython.display import set_matplotlib_formats

from matplotlib.colors import to_rgba
import seaborn as sns
sns.set()

#from tqdm.notebook import tqdm

from tqdm import tqdm

import torch

#print("Using torch", torch.__version__)

torch.manual_seed(42)

#torch.Tensor函数为张量创建内存地址

#torch.rand生成0-1区间上均匀分布的随机数
#torch.randn 生成标准高斯分布的随机数

#x = torch.rand(2, 3, 4)
y = torch.randn(2, 3, 4)
A = torch.tensor([[1, 3, 5], [2, 4, 6]])
print(A)

print(A.shape)