import os
import importlib
import numpy as np 

# Compile and load the cpp module
kernel = 'kernels/example.cpp'
cmd = f'g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) {kernel} -o example$(python3-config --extension-suffix)'
os.system(cmd)
my_package = importlib.import_module('example')
c = my_package.add(1, 2)
print(c)