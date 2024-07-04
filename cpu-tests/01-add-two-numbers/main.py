import os
import importlib

# 假定刚才的C++文件叫example.cpp
kernel = 'example.cpp'  
# 编译命令
cmd = f'g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) {kernel} -o example$(python3-config --extension-suffix)'
# 执行编译命令
os.system(cmd)
# 导入生成的example模块
my_package = importlib.import_module('example')
# 调用其add函数
c = my_package.add(1, 2)
# 打印输出结果（应为3）
print(c)
