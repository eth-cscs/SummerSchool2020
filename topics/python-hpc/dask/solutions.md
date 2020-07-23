# Solutions to the questions

* [02-exercise-cityblock-distance-matrix-scipy.dask](02-exercise-cityblock-distance-matrix-scipy.dask.ipynb)

1. Why is relevant for this implementation the fact that `scipy.spatial.distance.cdist` is not multithreaded?

 The cpu has a limit of threads. It will have to share the them with the `scipy.spatial.distance.cdist` function and with dask. Also, if the function is already mutithreaded probably we woudn't need to implement the parallelization with dask ;)
 
 
* [03-dask-arrays](03-dask-arrays.ipynb)

1. Let's consider now the operation `x.dot(x)`. Could you explain the results of the timings?

 Dask is not a replacement for numpy for every situation. The block-wise operations add overhead, we respect to their numpy counterparts. If we visualize the graph for the `x.dot(x)`, we can see that's pretty complicated, much more than `x.mean`'s.
 
* [06-dask-processes-vs-threads](06-dask-processes-vs-threads.ipynb)

 The function `euclidean_distance_matrix` doesn't release the GIL. As a result the graph executed using threads can not execute both tasks in parallel. Instead they will alternate running/blocking by aquiring and releasing the GIL, and the total run time is then the same as if ran serially.
 
 When we run the graph with processes, the tasks do run in parallel. This is because when using processes, in contrast to threads that share memory, the memory is copied by every rank. In this case the GIL is not involved and the to task can run in parallel.