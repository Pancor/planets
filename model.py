import datetime

import tensorflow as tf
from utils.confusion_matrix_callback import ConfusionMatrixCallback


def convert_av_training_set(av_training_set):
    label = av_training_set.numpy()
    if label == b'PC':
        number_value = 1
    elif label == b'AFP':
        number_value = 0
    elif label == b'NTP':
        number_value = 0
    else:
        error_msg = "Provided av_training_set : " + label.decode() + " is not supported."
        raise ValueError(error_msg)

    return tf.constant(number_value, dtype=tf.int8)


def read_tfrecord(record):
    features = {
        "local_view": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        "av_training_set": tf.io.FixedLenFeature([], tf.string, default_value=''),
    }
    example = tf.io.parse_single_example(record, features)
    local_light_curve = example["local_view"]
    av_training_set = example["av_training_set"]
    av_training_set = convert_av_training_set(av_training_set)

    tf.ensure_shape(local_light_curve, [201])
    tf.ensure_shape(av_training_set, [])

    return local_light_curve, av_training_set


train_files = tf.io.gfile.glob("./tfrecord/train-*")
train_dataset = tf.data.TFRecordDataset(filenames=train_files)
train_dataset = train_dataset.map(lambda x: tf.py_function(read_tfrecord, [x], (tf.float32, tf.int8)))
train_dataset = train_dataset.batch(32)

valid_files = tf.io.gfile.glob("./tfrecord/val-*")
valid_dataset = tf.data.TFRecordDataset(filenames=valid_files)
valid_dataset = valid_dataset.map(lambda x: tf.py_function(read_tfrecord, [x], (tf.float32, tf.int8)))
valid_dataset = valid_dataset.batch(32)

log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
file_writer_cm = tf.summary.create_file_writer(log_dir + "/cm")

model = tf.keras.Sequential([
    tf.keras.layers.Reshape(target_shape=(201, 1), input_shape=(201,)),
    tf.keras.layers.Conv1D(16, 3, activation='relu'),
    tf.keras.layers.MaxPool1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(48, activation="relu"),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.RMSprop(),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.Accuracy()]
)

confusion_matrix_callback = ConfusionMatrixCallback(log_dir, valid_dataset)
confusion_matrix_callback.set_showing_confusion_matrix(True)
model.fit(
    x=train_dataset,
    epochs=15,
    validation_data=valid_dataset,
    callbacks=[
        # tensorboard_callback,
        confusion_matrix_callback
    ],
)
