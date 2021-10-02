import datetime

import tensorflow as tf
import matplotlib.pyplot as plt


def convert_av_training_set(av_training_set):
    label = av_training_set.numpy()
    if label == b'PC':
        number_value = 1
    elif label == b'AFP':
        number_value = 2
    elif label == b'NTP':
        number_value = 3
    elif label == b'UNK':
        number_value = 4
    else:
        error_msg = "Provided av_training_set : " + label + " is not supported."
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

for parsed_record in valid_dataset.take(1):
    local_view = parsed_record[0][0].numpy()
    plt.plot(local_view, ".")
    plt.show()

log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(48, activation='relu', input_shape=(201,)),
#     tf.keras.layers.Dense(4, activation='softmax')
# ])

model = tf.keras.Sequential([
    tf.keras.layers.Reshape(target_shape=(201, 1), input_shape=(201,)),
    tf.keras.layers.Conv1D(16, 3, activation='relu'),
    tf.keras.layers.MaxPool1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(48, activation="relu"),
    tf.keras.layers.Dense(4, activation='softmax')
])
model.summary()

model.compile(optimizer='rmsprop',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['sparse_categorical_accuracy'])

model.fit(train_dataset,
          epochs=10,
          validation_data=valid_dataset,
          callbacks=[tensorboard_callback])
