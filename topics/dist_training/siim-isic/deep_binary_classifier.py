import os
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
import efficientnet.tfkeras as efn

def get_model(cfg, backbone, drop_rate=0.3, **kwargs):
    ''' create typical transfer learning model with a given backbone '''
    model_input = tf.keras.Input(shape=(cfg['net_size'], cfg['net_size'], 3), name='in')
    outputs = []

    x = backbone(include_top=False,
                 input_shape=(cfg['net_size'], cfg['net_size'], 3),
                 **kwargs)(model_input)

    x = tf.keras.layers.Dropout(rate=drop_rate, name='top_dropout')(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(model_input, x, name='EfficientNet')

    model.summary()
    return model

def compile_new_model(cfg,
                      backbone=efn.EfficientNetB0,
                      weights='noisy-student',
                      **kwargs):
    ''' create and compile a binary classifier '''
    model = get_model(cfg,
                      backbone=backbone,
                      weights=weights,
                      **kwargs)
    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=cfg['label_smooth_fac'])

    model.compile(
        optimizer = cfg['optimizer'],
        loss      = loss,
        metrics   = [tf.keras.metrics.AUC(name='auc')])

    return model
