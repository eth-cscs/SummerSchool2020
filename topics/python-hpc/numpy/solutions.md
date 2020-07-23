# Solution to questions

[01-numpy-array-internals](01-numpy-array-internals.ipynb)

1. How do you explain the next result? Is the result the same when using `x.flatten()` instead of `x.ravel()`?
 
 Both functions flatten the array. `x.flatten()` creates a new array with a new data buffer while `x.ravel()` changes the metadata.

2. The next three cells do the same two operations: transposing a matrix and flattening it. How do you explain the difference in execution time?

 From the previous answer you know that `x.flatten()` will be slower than `x.ravel()` since it allocates new memory.

 In the third case
```python
x.T.ravel()
```
 is not possible to flatten the array in the trasposed order by only change the metadata. As a result, `ravel()`, needs to create a new data buffer. For instance, consider the array
```python
0 1
2 3
```
with strides `(16, 8)`. Changing the strides to `(8,)` gives, `[0, 1, 2, 3]`. No new memory is needed. However, since the `ravel()` is applied to the trasposed matrix, it's not possible to get the array `[0 2, 1, 3]` by only changing the metadata.



* [02-broadcasting](02-broadcasting.ipynb)

1. What's the difference between shapes `(n,)`, `(1, n)` and `(n, 1)`?

 * `(n,)` is the same as `(1, n)`. A matrix with a single row and `n` columns.
 * `(n, 1)` is a matrix with a single column and `n` rows.


2. From what you have learned about the `numpy.ndarray`s: Does the operation `x[:, np.newaxis]` allocate new memory or could it be performed by only changing the metadata?

 Yes. No need to create new memory.
 
* [03-euclidean-distance-matrix-numpy](03-euclidean-distance-matrix-numpy.ipynb)

1. At this point you are starting to get acquainted with the `numpy.ndarray`s and it's memory managment. Could you analyse advantages and possible drawbacks of the `euclidean_broadcast` function? Write a positive and a negative point about it.

 **Advantages**:
 The function is fully vectorized. There are no loops.
 
 **Drawbacks**:
 `diff` will be a large matrix and allocating such large matrix can be slow. This is, comparing to the `euclidean_trick` implementation. Also the operations in the function are not multithreaded.