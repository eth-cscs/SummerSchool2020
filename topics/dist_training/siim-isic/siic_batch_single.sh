#!/bin/bash -l

#SBATCH --job-name=ss-siic
## #SBATCH --reservation=summer_school
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --constraint=gpu

module load daint-gpu
. /apps/daint/UES/6.0.UP04/sandboxes/sarafael/miniconda-ss2020/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Environment variables needed by the NCCL backend
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

export NCCL_IB_HCA=ipogif0
export NCCL_IB_CUDA_SUPPORT=1

srun nproc
srun which python
srun python -u Melanoma20-EffNetB7ns-Single.py
