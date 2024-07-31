import os
import importlib
import numpy as np
from timeit import timeit, repeat
import numba

# 编译pybind11模块并动态导入
name = 'mv'
cmd = f'g++ -O3 -mavx2 -Wall -shared -std=c++11 -fPIC -fopenmp $(python3 -m pybind11 --includes) {name}.cpp -o {name}$(python3-config --extension-suffix)'
ret = os.system(cmd)
if ret != 0:
    print('compiling cpp file failed')
my_package = importlib.import_module(name)

# 使用timeit.repeat进行性能测试的函数
def mybench(stmts, globals, n_warmup=10, n_iters=100):
    times = repeat(stmts, globals=globals, number=10, repeat=n_iters)
    return np.mean(times[n_warmup:]) / 10

# 使用numba并行地生成比直接np.random.randn(M, N)更快
@numba.njit(parallel=True)
def gen_matrix(M, N, dtype=np.float64):
    out = np.empty((M, N), dtype=dtype)
    for i in numba.prange(M):
        out[i] = np.random.randn(N)
    return out

for N in [1024*1024//2, 1024*1024, 1024*1024*2, 1024*1024*4, 1024*1024*8]:
    M = 512 * 2
    # 生成输入数组a和b
    a = gen_matrix(M, N)
    #a = np.random.randn(M, N)
    b = np.random.randn(N)
    print(f'input matrix shape: {M} x {N}')

    # 调用add函数并使用allclose确认结果正确
    c = my_package.kernel(a, b)
    assert np.allclose(c, a@b)

    # 性能测试
    t0 = mybench("my_package.kernel0(a, b)", globals())
    t1 = mybench("my_package.kernel(a, b)", globals())
    t2 = mybench("a @ b", globals())
    print(t0, t1, t2)
    print(f'tiled speedup over untiled: {(t0/t1):.4f}')
    print(f'tiled speedup over numpy: {(t2/t1):.4f}\n')
