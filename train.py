import numpy as np

import exofinder.datasets
import exofinder.models
import exofinder.utils.weights
import tensorflow as tf
import datetime
from exofinder.utils.confusion_matrix_callback import ConfusionMatrixCallback


def set_shapes(x, y, sample_weights):
    return x, y, tf.reshape(sample_weights, (len(sample_weights),))


local_train_dataset = exofinder.datasets.get_flux_series(path_to_files="./tfrecord/train-*")
local_valid_dataset = exofinder.datasets.get_flux_series(path_to_files="./tfrecord/val-*")

global_train_dataset = exofinder.datasets.get_flux_series(path_to_files="./tfrecord/train-*",
                                                          flux_type="global")
global_valid_dataset = exofinder.datasets.get_flux_series(path_to_files="./tfrecord/val-*",
                                                          flux_type="global")

weights = exofinder.utils.weights.calculate_weights(dataset=local_train_dataset)
local_train_dataset = exofinder.utils.weights.add_weights(dataset=local_train_dataset, weights=weights)
local_valid_dataset = exofinder.utils.weights.add_weights(dataset=local_valid_dataset, weights=weights)

local_view_model = exofinder.models.get_local_view_model()
global_view_model = exofinder.models.get_global_view_model()

combined_input = np.concatenate([local_view_model.output, global_view_model.output])

common_layers = tf.keras.layers.Dense(4, activation=tf.nn.relu)(combined_input)
common_layers = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(common_layers)
final_model = tf.keras.Model(inputs=[local_view_model.input, global_view_model.input], output=common_layers)

local_view_model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(),
    ],
)

log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
file_writer_cm = tf.summary.create_file_writer(log_dir + "/cm")

confusion_matrix_callback = ConfusionMatrixCallback(log_dir, local_valid_dataset)
# confusion_matrix_callback.set_showing_confusion_matrix(True)

local_train_dataset = local_train_dataset.batch(32)
local_valid_dataset = local_valid_dataset.batch(32)
local_train_dataset = local_train_dataset.map(lambda x, y, weight: (x, y, tf.reshape(weight, (len(weight),))))
local_valid_dataset = local_valid_dataset.map(set_shapes)

local_view_model.fit(
    x=local_train_dataset,
    epochs=50,
    validation_data=local_valid_dataset,
    callbacks=[
        tensorboard_callback,
        confusion_matrix_callback
    ],
)