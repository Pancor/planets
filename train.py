import numpy as np

import exofinder.datasets
import exofinder.models
import exofinder.utils.weights
import tensorflow as tf
import datetime
from exofinder.utils.confusion_matrix_callback import ConfusionMatrixCallback


local_train_dataset = exofinder.datasets.get_flux_series(path_to_files="./tfrecord/train-*")
global_train_dataset = exofinder.datasets.get_flux_series(path_to_files="./tfrecord/train-*",
                                                          flux_type="global")
local_valid_dataset = exofinder.datasets.get_flux_series(path_to_files="./tfrecord/val-*")
global_valid_dataset = exofinder.datasets.get_flux_series(path_to_files="./tfrecord/val-*",
                                                          flux_type="global")

# weights = exofinder.utils.weights.calculate_weights(dataset=local_train_dataset)
weights = np.array([0.64932, 2.17427])
local_train_dataset = exofinder.utils.weights.add_weights(dataset=local_train_dataset, weights=weights)

final_dataset = tf.data.Dataset.zip((local_train_dataset, global_train_dataset))
final_dataset = final_dataset.map(exofinder.datasets.convert_test_dataset_record)
final_dataset = final_dataset.batch(32)
final_dataset = final_dataset.map(exofinder.datasets.set_shape_for_weight)

valid_dataset = tf.data.Dataset.zip((local_valid_dataset, global_valid_dataset))
valid_dataset = valid_dataset.map(exofinder.datasets.convert_valid_dataset_record)
valid_dataset = valid_dataset.batch(32)

local_view_model = exofinder.models.get_local_view_model()
global_view_model = exofinder.models.get_global_view_model()
combined_input = tf.keras.layers.concatenate([local_view_model.output, global_view_model.output])

common_layers = tf.keras.layers.Dense(4, activation=tf.nn.relu)(combined_input)
common_layers = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(common_layers)
final_model = tf.keras.Model(inputs=[local_view_model.input, global_view_model.input], outputs=common_layers)

final_model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(),
    ],
)

log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
file_writer_cm = tf.summary.create_file_writer(log_dir + "/cm")

confusion_matrix_callback = ConfusionMatrixCallback(log_dir, valid_dataset)
# confusion_matrix_callback.set_showing_confusion_matrix(True)

final_model.fit(
    x=final_dataset,
    epochs=50,
    validation_data=valid_dataset,
    callbacks=[
        tensorboard_callback,
        confusion_matrix_callback
    ],
)
