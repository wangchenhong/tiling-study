import os
import importlib
import numpy as np
from timeit import timeit, repeat

# 编译pybind11模块并动态导入
name = 'vec_add'
cmd = f'g++ -O3 -Wall -shared -std=c++11 -fPIC -fopenmp $(python3 -m pybind11 --includes) {name}.cpp -o {name}$(python3-config --extension-suffix)'
# cmd = f'g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) {name}.cpp -o {name}$(python3-config --extension-suffix)'
ret = os.system(cmd)
if ret != 0:
    print('compiling cpp file failed')
my_package = importlib.import_module(name)

# 生成输入数组a和b
N = 200000
a = np.random.randn(N)
b = np.random.randn(N)

# 调用add函数并使用allclose确认结果正确
c = my_package.add(a, b)
assert np.allclose(c, a+b)

# 性能测试
t0 = repeat("my_package.add(a, b)", globals=globals(), number=10, repeat=100)
t1 = repeat("a + b", globals=globals(), number=10, repeat=100)
#print(t0, t1)
print(np.mean(t0[10:]) / 10, np.mean(t1[10:]) / 10)