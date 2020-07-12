# Serial miniapp implementation

To compile and run the serial version of the miniapp

```
# to build
module load daint-gpu
module swap PrgEnv-cray PrgEnv-gnu
make -j4

# to run without an allocation
srun -Cgpu -n1 -c12 --hint=nomultithread --reservation=summer_school ./main 128 128 128 0.01

# to run with an interactive session
salloc -Cgpu --reservation=summer_school
srun -n1 -c12 --hint=nomultithread ./main 128 128 128 0.01

# to plot
module load PyExtensions/python3-CrayGNU-19.10
python3 plotting.py
```

Benchmark results on Piz Daint, measured in CG iterations/second:

```
           -------------------------------------------------------------
           | 1 thread | 2 threads | 4 threads | 8 threads | 12 threads |
------------------------------------------------------------------------
|  64x 64  | ?        | ?         | ?         | ?         | ?          |
| 256x256  | ?        | ?         | ?         | ?         | ?          |
| 512x512  | ?        | ?         | ?         | ?         | ?          |
------------------------------------------------------------------------

```
