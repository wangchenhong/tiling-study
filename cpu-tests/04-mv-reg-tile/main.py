import os
import importlib
import numpy as np
from timeit import timeit, repeat
import time
import numba

# 编译pybind11模块并动态导入
name = 'mv'
cmd = f'g++ -O3 -mavx2 -Wall -shared -std=c++11 -fPIC -fopenmp $(python3 -m pybind11 --includes) {name}.cpp -o {name}$(python3-config --extension-suffix)'
ret = os.system(cmd)
if ret != 0:
    print('compiling cpp file failed')
my_package = importlib.import_module(name)

import torch.utils.benchmark as torchbench

def bench(fn):
    t0 = torchbench.Timer(
        stmt='fn()',
        globals={'fn': fn},
    )
    return t0.blocked_autorange(min_run_time=1).mean

def mybench(fn, clear_cache=True, n_warmup=10, n_iters=500):
    times = []
    for i in range(n_iters+n_warmup):
        if clear_cache:
            _ = np.zeros(256*1024*1024//8, dtype=np.float64)
        t0 = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
    return np.mean(times[n_warmup:])

@numba.njit(parallel=True)
def gen_matrix(M, N, dtype=np.float64):
    out = np.empty((M, N), dtype=dtype)
    for i in numba.prange(M):
        out[i] = np.random.randn(N)
    return out

for N in [1024*1024*2, 1024*1024*1, 1024*1024//2]:
#for N in [1024*1024*8, 1024*1024*4, 1024*1024*2, 1024*1024*1, 1024*1024//2]:
    M = 512
    # 生成输入数组a和b
    a = gen_matrix(M, N)
    #a = np.random.randn(M, N)
    b = np.random.randn(N)
    print('done generating inputs')

    # 调用add函数并使用allclose确认结果正确
    c = my_package.kernel(a, b)
    assert np.allclose(c, a@b)

    # 性能测试
    t0 = repeat("my_package.kernel0(a, b)", globals=globals(), number=10, repeat=100)
    t1 = repeat("my_package.kernel(a, b)", globals=globals(), number=10, repeat=100)
    print(t0, t1)
    print(np.mean(t0[20:]), np.mean(t1[20:]))

    t0 = bench(lambda: my_package.kernel0(a, b))
    t1 = bench(lambda: my_package.kernel(a, b))
    t2 = bench(lambda: a @ b)
    print(t0, t1, t2)
    print(f'speedup with tiling: {(t0/t1):.4f}\n')

    t0 = mybench(lambda: my_package.kernel0(a, b))
    t1 = mybench(lambda: my_package.kernel(a, b))
    t2 = mybench(lambda: a @ b)
    print(t0, t1, t2)
    print(f'speedup with tiling: {(t0/t1):.4f}\n')
    print()
    # print(f'speedup with tiling: {(t0[0]/t1[0]):.4f}\n')



    
