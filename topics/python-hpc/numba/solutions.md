# Solutions to teh questions

* [cityblock-distance-matrix-numba.jit](cityblock-distance-matrix-numba.jit.ipynb)

1. How do you explain the difference in execution times?

 The python function is the slowest. We know that using such long loops is slow.
 
 `cityblock_numba1` is faster than `cityblock_numba2` since it uses operations which numba knows how to optimize. `cityblock_numba2` calls the numpy function `np.linalg.norm` which is C code, so numba can not compile it.