import numpy as np
import sklearn.utils
import tensorflow as tf


def calculate_weights(dataset):
    labels = [y for x, y in dataset.as_numpy_iterator()]
    return sklearn.utils.class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(labels),
                                                           y=labels)


def add_weights(dataset, weights):
    tensor_weights = tf.convert_to_tensor(weights, tf.float32)
    return dataset.map(lambda x, y: tf.py_function(__add_weight, [x, y, tensor_weights],
                                                   (tf.float32, tf.int8, tf.float32)))


def __add_weight(curve, label, weights):
    weight = weights[label.numpy()]
    return curve, label, tf.constant(weight, dtype=tf.float32)
