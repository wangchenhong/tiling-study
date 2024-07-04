import os
import importlib
import numpy as np
from timeit import timeit

# Compile and load the cpp module
name = 'vec_add'
kernel = 'kernels/vec_add.cpp'
cmd = f'g++ -O3 -Wall -shared -std=c++11 -fPIC -fopenmp $(python3 -m pybind11 --includes) {kernel} -o {name}$(python3-config --extension-suffix)'
ret = os.system(cmd)
if ret != 0:
    print('compile cpp file failed')

my_package = importlib.import_module(name)
N = 200000
a = np.random.randn(N)
b = np.random.randn(N)
c = my_package.add(a, b)
assert np.allclose(c, a+b)

t0 = timeit("my_package.add(a, b)", globals=globals(), number=50)
t1 = timeit("a + b", globals=globals(), number=50)
print(t0, t1)