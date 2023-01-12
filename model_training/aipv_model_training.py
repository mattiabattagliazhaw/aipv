# -*- coding: utf-8 -*-
import os
import pandas as pd
import tensorflow as tf
import pickle
import numpy as np
from tensorflow import keras
import keras.backend as K
from keras.models import Model
from wandb.keras import WandbCallback
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers import *
from aipv_wandb_callback_2D_regions import AipvWandCallbackUncertainty
import wandb
import subprocess

subprocess.call("wandb login 75b2f297e9f1c53c2f60e3bb18137309a1d8d492", shell=True)

hyperparameter_defaults = dict(optimizer="adam",
                               pool=2,
                               dropout=0.29,
                               dense_num=4096,
                               loss="neg_log_likelihood",
                               dataset="example_data",
                               n_samples=None,
                               targets=['V2','ln_r_sheet','log_dark_saturation_current','ln_rho_par'],
                               batch_size=512,
                               learning_rate=2.3e-5,
                               model='VGG19',
                               bagging_seed=1
                               )

wandb.init(entity="ai-pv", project="ai-pv-uncertainty2", config=hyperparameter_defaults)
data_root = r"C:\Users\batg\PycharmProjects\aipv/"

data_path = data_root + wandb.config.dataset

np.random.seed(wandb.config.bagging_seed)
tf.random.set_seed(wandb.config.bagging_seed)

def eluPlus1(x):
    return K.elu(x)+1

keras.utils.get_custom_objects().update({'elu_plus_1': Activation(eluPlus1)})

def log_sum_exp(x, axis=1):
    """Log-sum-exp trick implementation"""
    x_max = K.max(x, axis=axis, keepdims=True)
    return K.log(K.sum(K.exp(x - x_max),
                       axis=axis, keepdims=True))+x_max

def neg_log_likelihood(y_true,y_pred):
    n = len(wandb.config.targets)
    mu = y_pred[:,:n]
    mu = K.clip(mu,-1,1)
    sig2 = y_pred[:, n:]
    sig2 = K.clip(sig2,1e-4,1)
    exponent = K.mean(K.log(sig2)/2,axis=1)+K.mean(K.square(y_true[:,:n]-mu)/(2*sig2),axis=1)
    return K.mean(exponent)

def load_train_randomized():
    x_train = np.load(os.path.join(data_path, 'x_train.npy'), allow_pickle=True)
    x_train[np.isneginf(x_train)] = 0
    x_train[np.isinf(x_train)] = 0
    x_train[:,:,:,0]=0
    print(pd.read_csv(os.path.join(data_path, 'y_train.txt')).columns)
    y_train = pd.read_csv(os.path.join(data_path, 'y_train.txt'))[wandb.config.targets]
    x_test = np.load(os.path.join(data_path, 'x_test.npy'), allow_pickle=True)
    x_test[np.isneginf(x_test)] = 0
    x_test[np.isinf(x_test)] = 0
    x_test[:, :, :, 0] = 0
    y_test = pd.read_csv(os.path.join(data_path, 'y_test.txt'))[wandb.config.targets]

    y_test = y_test.values
    if True:
        y_valid = y_test[0:3]
        x_valid = x_test[0:3]

        all_data = np.vstack([x_train,x_test[3:]])
        all_target = np.vstack([y_train.values, y_test[3:]])
        x_ids = list(range(len(all_data)))

        np.random.shuffle(np.array(x_ids))
        all_data = all_data[x_ids][:int(0.75*len(all_data))]
        all_target = all_target[x_ids][:int(0.75*len(all_target))]
        split_length = int(len(all_data)*0.04)
        split = int(len(all_data)*wandb.config.bagging_seed/20)

        x_train = np.vstack([all_data[:split],all_data[split+split_length:]])
        y_train = np.vstack([all_target[:split],all_target[split + split_length:]])
        x_test = np.vstack([x_valid,all_data[split:split+split_length]])
        y_test = np.vstack([y_valid,all_target[split:split+split_length]])


    filter_train = np.count_nonzero(x_train[:, :, :, 2], axis=(1, 2)) > 0
    filter_test = np.count_nonzero(x_test[:, :, :, 2], axis=(1, 2)) > 0
    y_train_filtered = y_train[filter_train]
    y_test_filtered = y_test[filter_test]
    y_train_filtered = np.hstack([y_train_filtered,np.zeros([len(y_train_filtered),len(wandb.config.targets)])])
    y_test_filtered = np.hstack([y_test_filtered, np.zeros([len(y_test_filtered), len(wandb.config.targets)])])
    return np.nan_to_num(np.array(x_train[filter_train])), y_train_filtered,np.nan_to_num(np.array(x_test[filter_test])), y_test_filtered


def read_and_normalize_train_data():
    train_data, train_target,test_data, test_target = load_train_randomized()
    train_data = np.nan_to_num(np.array(train_data, dtype=np.float32))
    train_target = np.nan_to_num(np.array(train_target, dtype=np.float32))
    test_data = np.nan_to_num(np.array(test_data, dtype=np.float32))
    test_target = np.nan_to_num(np.array(test_target, dtype=np.float32))

    assert not np.any(np.isnan(train_data))
    assert not np.any(np.isnan(train_target))
    assert not np.any(np.isnan(test_data))
    assert not np.any(np.isnan(test_target))

    m=np.array([0,0.6,0])
    s=np.array([1,0.05,1])

    train_data -= m
    train_data /= s
    test_data -= m
    test_data /= s

    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')

    return train_data, train_target,test_data, test_target,[m,s]


def adapt_vgg16(image_size) -> Model:
    """This code uses adapts vgg16 or vgg19  with to a regression
    problem and adds the top for neg_log_likelihood loss.
    Most of this code is adapted from the official keras documentation.

    Returns
    -------
    Model
        The keras model.
    """
    image_input = keras.layers.Input(
        shape=image_size
    )  # input shapes of the images should always be 224x224x3 with EfficientNetB0
    # use the downloaded and converted newest EfficientNet wheights
    if wandb.config.model == 'VGG16':
        model = keras.applications.VGG16(include_top=False,weights=None, input_tensor=image_input,pooling=None)
    else:
        model = keras.applications.VGG19(include_top=False, weights=None, input_tensor=image_input, pooling=None)
    # Freeze the pretrained weights
    #model.trainable = False

    # Rebuild top

    x = keras.layers.Flatten()(model.output)
    x = keras.layers.Dense(wandb.config.dense_num, activation='relu')(x)
    x = keras.layers.Dropout(wandb.config.dropout)(x)
    x = keras.layers.Dense(wandb.config.dense_num, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(wandb.config.dropout)(x)

    branchA = keras.layers.Dense(len(wandb.config.targets), activation="linear", name="pred1")(x)
    branchB = keras.layers.Dense(len(wandb.config.targets), activation="elu_plus_1", name="pred2")(x)
    predictions = keras.layers.concatenate([branchA, branchB])

    model = keras.models.Model(inputs=image_input, outputs=predictions, name="EfficientNet")
    if wandb.config.learning_rate == 'exp_decay':
        learning_rate = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3,
            decay_steps=100000,
            decay_rate=0.95)
    else:
        learning_rate = wandb.config.learning_rate

    if wandb.config.optimizer == 'adam':
        opt = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=0.1)
    else:
        opt = wandb.config.optimizer
    if wandb.config.loss == 'neg_log_likelihood':
        model.compile(loss=neg_log_likelihood, optimizer=opt)
    else:
        model.compile(loss=wandb.config.loss, optimizer=opt)
    return model

def train_model(batch_size=32, nb_epoch=20):
    stoppercallback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

    scaler = MinMaxScaler(feature_range=(0.001, 0.999))

    train_data, train_target,test_data, test_target,image_scaler = read_and_normalize_train_data()

    scaler_array=np.hstack((train_target[:,:len(wandb.config.targets)],train_target[:,:len(wandb.config.targets)]))
    scaler.fit(scaler_array)
    train_target = scaler.transform(train_target)
    test_target = scaler.transform(test_target)

    x_train = train_data
    x_valid = test_data
    y_train = train_target
    y_valid = test_target

    image_size = np.shape(x_train[0])


    model = adapt_vgg16(image_size)


    with open(os.path.join(data_root, "measurements//test_measurement_regions.pckl"), 'rb') as file:
        test_measurement = np.array(pickle.load(file))
    test_measurement[:,:,:,0]=0

    aipv_callback = AipvWandCallbackUncertainty(data_path, scaler, step=20,
                                                training_data=(x_train, y_train),
                                                validation_data=(x_valid, y_valid),
                                                validation_measurement=test_measurement,
                                                target_names=wandb.config.targets, image_scaler=image_scaler,
                                                run_name=wandb.run.name)

    test_measurement -= image_scaler[0]
    test_measurement /= image_scaler[1]

    model.fit(x_train[:], y_train,shuffle=True, batch_size=batch_size, epochs=nb_epoch, verbose=1,
              validation_data=(x_valid, y_valid),
              callbacks=[WandbCallback(save_model=True), stoppercallback, aipv_callback])


if __name__ == '__main__':
    train_model(nb_epoch=500, batch_size=wandb.config.batch_size)
