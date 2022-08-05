import numpy as np

import exofinder.datasets
import exofinder.models
import exofinder.utils.weights
import tensorflow as tf
import datetime
from exofinder.utils.confusion_matrix_callback import ConfusionMatrixCallback


def set_shapes(x, y, sample_weights):
    return x, y, tf.reshape(sample_weights, (len(sample_weights),))


def test(x, y):
    return {
        "local_flux_input": x[0],
        "global_flux_input": y
    }, x[1], x[2]


def test_valid(x, y):
    return {
               "local_flux_input": x[0],
               "global_flux_input": y
           }, x[1]

local_train_dataset = exofinder.datasets.get_flux_series(path_to_files="./tfrecord/train-*")
global_train_dataset = exofinder.datasets.get_flux_series(path_to_files="./tfrecord/train-*",
                                                          flux_type="global")
local_valid_dataset = exofinder.datasets.get_flux_series(path_to_files="./tfrecord/val-*")
global_valid_dataset = exofinder.datasets.get_flux_series(path_to_files="./tfrecord/val-*",
                                                          flux_type="global")

#weights = exofinder.utils.weights.calculate_weights(dataset=local_train_dataset)
weights = np.array([0.64932, 2.17427])
local_train_dataset = exofinder.utils.weights.add_weights(dataset=local_train_dataset, weights=weights)

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

#local_train_dataset = local_train_dataset.batch(32)
#local_train_dataset = local_train_dataset.map(lambda x, y, weight: (x, y, tf.reshape(weight, (len(weight),))))
#
# global_train_dataset = global_train_dataset.batch(32)

final_dataset = tf.data.Dataset.zip((local_train_dataset, global_train_dataset))
final_dataset = final_dataset.map(test)

# for element in final_dataset.as_numpy_iterator():
#     print(element)
#     break

# final_dataset = final_dataset.map(map_func=lambda x, y: new_py_function(
#     func=test, inp=[x, y], Tout=({"local_flux_input": tf.float32, "global_flux_input": tf.float32}, tf.int8, tf.float32)
# ))
final_dataset = final_dataset.batch(32)
final_dataset = final_dataset.map(lambda x, y, weight: (x, y, tf.reshape(weight, (len(weight),))))

valid_dataset = tf.data.Dataset.zip((local_valid_dataset, global_valid_dataset))
valid_dataset = valid_dataset.map(test_valid)
valid_dataset = valid_dataset.batch(32)

# final_model.summary()

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
