#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import gc
import math
import time
import random
random.seed(42)

import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid

import PIL

from tqdm import tqdm
from attrdict import AttrDict

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K

import efficientnet.tfkeras as efn

from one_cycle_scheduler import *


# In[2]:


num_workers = int(os.environ['SLURM_NNODES'])
node_id = int(os.environ['SLURM_NODEID'])

fold  = 4
kfold = 5
debug = True
batch_size = # TODO: scale your global batch_size
             # note: keras always expects global batch sizes
             #       and splits it between the nodes accordingly


# In[3]:


CFG = AttrDict(
    batch_size       = batch_size,

    # Image sizes
    read_size        = 256, 
    crop_size        = 224, 
    net_size         = 224, 

    # Training Schedule
    LR_START         = 3e-3 * batch_size,
    LR_MAX           = 3e-2 * batch_size,
    LR_END           = 1e-2 * batch_size,
    LR_RAMPUP_EPOCHS = 2 if debug else 0.5,
    epochs           = 6 if debug else 16,

    # Image Augmentation
    rot              = 180.0,
    shr              =   2.0,
    hzoom            =   8.0,
    wzoom            =   8.0,
    hshift           =   8.0,
    wshift           =   8.0,
    tta_steps        = 2 if debug else 4,

    optimizer        = tfa.optimizers.SGDW(lr=4e-2 * batch_size,
                                            nesterov=True,
                                            weight_decay=1e-5),
    label_smooth_fac = 0.05,
)

print(CFG)


# In[4]:


BASEPATH   = "/scratch/snx3000/dealmeih/summer_school/melanoma-256x256/"
OUTPUTPATH = os.environ['SCRATCH'] + f"/summer_school/melanoma_f{fold}/"
print(f"Training with {BASEPATH} and saving at {OUTPUTPATH}")

df_train = pd.read_csv(BASEPATH + 'train.csv')
df_test  = pd.read_csv(BASEPATH + 'test.csv')
df_sub   = pd.read_csv(BASEPATH + 'sample_submission.csv')

files_test  = np.sort(tf.io.gfile.glob(BASEPATH + 'test*.tfrec'))
files_train = np.sort(tf.io.gfile.glob(BASEPATH + 'train*.tfrec'))

idx = np.arange(len(files_train))
files_valid = files_train[idx % kfold == fold]
files_train = files_train[idx % kfold != fold]

print('Validation:\n', files_valid)
print('Training:\n',   files_train)


# In[5]:


# sample some data
pd.read_csv(BASEPATH + 'train.csv').sample(frac=1, random_state=4)


# In[6]:


# Disable greedy GPU memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

if num_workers == 1:
    strategy = tf.distribute.get_strategy() # default strategy
else:
    strategy = # TODO use MultiWorkerMirroredStrategy

replicas = strategy.num_replicas_in_sync
assert replicas == num_workers, f"replicas:{replicas} num_workers:{num_workers}"
print(replicas, gpus)


# In[7]:



# In[8]:


from siic_tfrec_dataset import get_dataset, count_data_items, show_dataset

ds_train = get_dataset(files_train, CFG, augment=True, shuffle=True, repeat=True)
ds_valid = get_dataset(files_valid, CFG, augment=False, shuffle=False, repeat=True)

steps_train = count_data_items(files_train) // CFG.batch_size
steps_valid = count_data_items(files_valid) // CFG.batch_size
if debug:
    steps_train = 50

print(f'Training for {steps_train} steps and ' +
      f'validating for {steps_valid} steps per epoch, '
      f'across {num_workers} node(s)')



# In[17]:


from deep_binary_classifier import compile_new_model

with strategy.scope():
    model = compile_new_model(CFG,
                              efn.EfficientNetB2,
                              BASEPATH + '../efficientnet-b2_noisy-student_notop.h5',
                              pooling='avg',
                             )

# EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
# https://arxiv.org/pdf/1905.11946.pdf


# In[18]:



# Log loss and metric history in TensorBoard format
tb_logdir = OUTPUTPATH + 'logs'
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=tb_logdir, update_freq=50,
                                                histogram_freq=0, profile_batch=0)

# In TF <=2.2 ModelCheckpoint allows fault tolerance
# in cases where workers die or are otherwise unstable.
# We do this by preserving training state in the distributed file system of your choice,
# such that upon restart of the instance that previously failed or preempted,
# the training state is recovered.
checkpoint_path = OUTPUTPATH + 'weights{epoch:03d}-auc{val_auc:.05f}.h5'
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=False,
    save_best_only=True,
    verbose=1, monitor='val_auc', mode='max',
)

# Stops training if metric dosen't improve for 'patience' epoches
stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_auc', min_delta=1e-5,
                                               mode='max', patience=CFG.epochs//2, verbose=1,
                                               restore_best_weights=True)

# LR and momentum scheduler
lr_sched_cb = OneCycleScheduler(lr_max=CFG.LR_MAX,
                                steps=CFG.epochs * steps_train,
                                mom_min=0.85, mom_max=0.95,
                                phase_1_pct=CFG.LR_RAMPUP_EPOCHS / CFG.epochs,
                                div_factor=CFG.LR_MAX / CFG.LR_START,
                                final_div_factor=CFG.LR_MAX / CFG.LR_END)

# lr_sched_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='auto', verbose=1,
#                                                    factor=0.3, patience=2, cooldown=1,
#                                                    min_delta=1e-4, min_lr=1e-8)

# Nicer progress output, both on jupyter and console
tqdm_progbar = tfa.callbacks.TQDMProgressBar()

callbacks = [tensorboard_cb, checkpoint_cb, stopping_cb, lr_sched_cb, tqdm_progbar]



# In[20]:


history = model.fit(ds_train, 
                    verbose          = 0, # use TQDMProgressBar instead
                    steps_per_epoch  = steps_train,
                    epochs           = CFG.epochs,
                    validation_data  = ds_valid,
                    validation_steps = steps_valid,
                    callbacks        = callbacks
                   )


# In[21]:


if hasattr(lr_sched_cb, 'plot'):
    lr_sched_cb.plot()

