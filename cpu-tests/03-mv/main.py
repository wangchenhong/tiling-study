import os
import importlib
import numpy as np
import time
from timeit import timeit, repeat

# 编译pybind11模块并动态导入
name = 'mv'
cmd = f'g++ -O3 -Wall -shared -std=c++11 -fPIC -fopenmp $(python3 -m pybind11 --includes) {name}.cpp -o {name}$(python3-config --extension-suffix)'
ret = os.system(cmd)
if ret != 0:
    print('compiling cpp file failed')
my_package = importlib.import_module(name)

# 使用timeit.repeat进行性能测试的函数
def mybench(stmts, globals, n_warmup=10, n_iters=100):
    times = repeat(stmts, globals=globals, number=10, repeat=n_iters)
    return np.mean(times[n_warmup:]) / 10

# 生成输入数组a和b
M = 1024
N = 1024*1024
a = np.random.randn(M, N)
b = np.random.randn(N)

# 调用add函数并使用allclose确认结果正确
c = my_package.kernel(a, b)
assert np.allclose(c, a@b)

# 性能测试
t0 = mybench("my_package.kernel(a, b)", globals())
t1 = mybench("a @ b", globals())
print(t0, t1)
print(f'speedup over numpy: {(t1/t0):.4f}\n')