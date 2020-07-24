import os
from test_tube import Experiment, HyperOptArgumentParser, SlurmCluster
from melanoma20_train import train
import tensorflow_addons as tfa


def run_experiment(hparams, *_):
    print(os.environ)
    num_workers = int(os.environ['SLURM_NNODES'])
    node_id = int(os.environ['SLURM_NODEID'])

    fold  = 0
    kfold = 5
    debug = True
    path = os.environ['SCRATCH'] + f"/summer_school/hopt{fold}/job" + os.environ['SLURM_TASK_PID'] + os.environ['HOSTNAME']
    print(node_id, path)

    exp = Experiment(save_dir=f'{path}/exp')
    exp.argparse(hparams)

    hparams.optimizer = tfa.optimizers.LAMB(lr=hparams.lr,
                                            weight_decay_rate=hparams.wd)
    print(hparams, flush=True)

    # start trainer
    auc = train(vars(hparams), num_workers, node_id, fold, kfold, debug, path)
    print(auc)

    # save Experiment
    exp.add_scalar('auc', auc)
    exp.save()


if __name__ == '__main__':
    # Set up our argparser and make the y_val tunable.
    parser = HyperOptArgumentParser(strategy='random_search')
    parser.add_argument('--test_tube_exp_name', default='tt_exp')
    parser.add_argument('--batch_size', default=80, type=int)
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--gpus', default=2, type=int)
    parser.add_argument('--log_path', default=os.environ['SCRATCH'] + '/summer_school/hs_log')
    parser.opt_range('--lr', default=2e-4, type=float, tunable=True, low=1e-4, high=1e-3, nb_samples=100, log_base=10)
    parser.opt_range('--wd', default=1e-5, type=float, tunable=True, low=1e-7, high=1e-4, nb_samples=100, log_base=10)
    hyperparams = parser.parse_args()

    # Enable cluster training.
    cluster = SlurmCluster(
        hyperparam_optimizer=hyperparams,
        log_path=hyperparams.log_path,
        python_cmd='python3',
    )

    # Email results if your hpc supports it.
#    cluster.notify_job_statusi(email='henrique.mendonca@cscs.ch', on_done=True, on_fail=True)

    # SLURM Module to load.
    cluster.load_modules([
        'daint-gpu',
    ])

    # Add commands to the non-SLURM portion.
    cluster.add_command('. /apps/daint/UES/6.0.UP04/sandboxes/sarafael/miniconda-ss2020/bin/activate')

    cluster.add_command('export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK')
    cluster.add_command('export NCCL_DEBUG=INFO')
    cluster.add_command('export PYTHONFAULTHANDLER=1')
    cluster.add_command('export NCCL_IB_HCA=ipogif0')
    cluster.add_command('export NCCL_IB_CUDA_SUPPORT=1')

    cluster.add_command('srun nproc')
    cluster.add_command('srun which python')

    # Add custom SLURM commands which show up as:
    # #comment
    # #SBATCH --cmd=value
    # ############
    cluster.add_slurm_cmd(cmd='constraint', value='gpu', comment='GPU nodes')
    cluster.add_slurm_cmd(cmd='cpus-per-task', value=24, comment='ncpus')
    cluster.add_slurm_cmd(cmd='ntasks-per-core', value=1, comment='ntasks-per-core')
    cluster.add_slurm_cmd(cmd='ntasks-per-node', value=1, comment='ntasks-per-node')


    # Set job compute details (this will apply PER set of hyperparameters.)
    cluster.per_experiment_nb_gpus = 1
    cluster.per_experiment_nb_nodes = hyperparams.gpus 
    cluster.memory_mb_per_node = 32*1024

    # Run hyperparameter combinations.
    cluster.optimize_parallel_cluster_gpu(
        # Function to execute:
        run_experiment,
        # Number of hyperparameter combinations to search:
        nb_trials=2,
        # This is what will display in the slurm queue:
        job_name='tt_job'
    )

