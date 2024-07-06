import os
import importlib
import numpy as np
from timeit import timeit

# 编译pybind11模块并动态导入
name = 'mv'
cmd = f'g++ -O3 -Wall -shared -std=c++11 -fPIC -fopenmp $(python3 -m pybind11 --includes) {name}.cpp -o {name}$(python3-config --extension-suffix)'
ret = os.system(cmd)
if ret != 0:
    print('compiling cpp file failed')
my_package = importlib.import_module(name)

# 生成输入数组a和b
M = 1024
N = 1024*1024
a = np.random.randn(M, N)
b = np.random.randn(N)

# 调用add函数并使用allclose确认结果正确
c = my_package.kernel(a, b)
assert np.allclose(c, a@b)

# 性能测试
t0 = timeit("my_package.kernel(a, b)", globals=globals(), number=10)
t1 = timeit("a @ b", globals=globals(), number=10)
print(t0, t1)