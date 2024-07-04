import subprocess
import importlib
import numpy as np 

# Compile and load the cpp module
cmd = '$ c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) add.cpp -o example$(python3-config --extension-suffix)'
subprocess.run(cmd, cwd='kernels/', check=True)
my_package = importlib.import('add')

for N in [16384, 16384*2]:
    a = np.random.randn(N//2, N)
    b = np.random.randn(N)
    print(a, b, a @ b)

