# Serial miniapp implementation

To compile and run the serial version of the miniapp

```
# to build
module load daint-gpu
module swap PrgEnv-cray PrgEnv-gnu
make -j4

# to run without an allocation
srun -Cgpu -n1 -c12 --hint=nomultithread --reservation=summer_school ./main 128 128 100 0.01

# to run with an interactive session
salloc -Cgpu --reservation=summer_school
srun -n1 -c12 --hint=nomultithread ./main 128 128 128 0.01

# to plot
module load PyExtensions/python3-CrayGNU-19.10
python3 plotting.py
```

Benchmark results on Piz Daint, measured in CG iterations/second, executed with:

```
OMP_NUM_THREADS=$nt srun -n1 -c12 --hint=nomultithread ./main $dim $dim 200 0.01
```

Where `$nt` is the number of threads, and `$dim` is the spatial dimsnsion.


```
           -------------------------------------------------------------
           | 1 thread | 2 threads | 4 threads | 8 threads | 12 threads |
------------------------------------------------------------------------
| 128x128  | 7222.0   | 12959.1   | 20225.1   | 26231.0   | 26095.7    |
| 256x256  | 1742.6   |  3250.2   |  6823.0   | 12021.4   | 15036.6    |
| 512x512  |  454.8   |  1047.8   |  1929.1   |  3409.4   |  4544.6    |
------------------------------------------------------------------------

```
