### 2. Distributed Training with tf.keras
Possible lecture contents for 2nd half
Fri 24.07 10.30-12:00 presented by Henrique

#### Motivation
- Batch size and PCA Visualizations

#### Distributed training with tf.keras 2
- [tf.distribute.Strategy](https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy)

- Multi-GPU:
```python
        strategy = [tf.distribute.MirroredStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy)()
        with strategy.scope():
            model = build_and_compile_model()
        model.fit(dataset, epochs steps_per_epoch)
```
- [NCCL](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html) AllReduce by default
Automatic data sharding across GPU's

#### Multi-worker/node distribution
- Additional complexity 
   - external network communication
   - separated OS
   - separated processes
       - Facilitated by ipc magic on jupyterlab

#### Multi-worker/node distribution
- [tf.distribute.experimental.MultiWorkerMirroredStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/MultiWorkerMirroredStrategy)
- Communication:
   - NCCL AllReduce for all-reduce (if available)
   - Ring algorithm for all-gather
- Cluster Resolver:
   - defaults to TfConfig
```python
        os.environ['TF_CONFIG'] = '{
            "cluster": {"worker": ["nid01111:8888", "nid02222:8888"]},
            "task": {"type": "worker", "index": "0"}
        }'
```

#### Multi-worker distribution with SLURM
- TensorFlow 2.2+
```python
    tf.distribute.cluster_resolver.SlurmClusterResolver(
        port_base=8888, auto_set_gpu=True, rpc_layer='grpc',
        jobs=None, gpus_per_node=None, gpus_per_task=None, tasks_per_node=None
    )
```
- All parameters are automatically queried from SLURM

#### Multi-worker distribution with SLURM
```python
    %%px
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        cluster_resolver=tf.distribute.cluster_resolver.SlurmClusterResolver(),
        communication=tf.distribute.experimental.CollectiveCommunication.NCCL,
    )
    with strategy.scope():
        model = build_and_compile_model()
    model.fit(dataset, epochs steps_per_epoch)
```

#### Batch Norm Synchronisation
- [tf.keras.layers.experimental.SyncBatchNormalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/SyncBatchNormalization)
- AllReduce across BN layers during forward pass
- Performance impact ? TODO

#### Scaling learning rate and momentum
- [https://arxiv.org/pdf/1706.02677.pdf](https://arxiv.org/pdf/1706.02677.pdf)
- [https://arxiv.org/pdf/1904.00962.pdf](https://arxiv.org/pdf/1904.00962.pdf)
- ...

#### Configure JupyterLab
#### Dataset
#### Assignment[s]
#### Overriding Keras training step (optional)
