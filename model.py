import datetime

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import itertools
import sklearn.metrics
import io


def convert_av_training_set(av_training_set):
    label = av_training_set.numpy()
    if label == b'PC':
        number_value = 1
    elif label == b'AFP':
        number_value = 2
    elif label == b'NTP':
        number_value = 3
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


def plot_confusion_matrix(cm, class_names):
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap("Blues"))
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')

    plt.close(figure)
    buf.seek(0)

    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image


def log_confusion_matrix(epoch, logs):
    valid_curves = []
    valid_labels = []

    for batch in valid_dataset:
        for curve in batch[0]:
            valid_curves.append(curve.numpy())
        for label in batch[1]:
            valid_labels.append(label.numpy())

    valid_curves = np.array(valid_curves)
    valid_labels = np.array(valid_labels)

    predictions = np.argmax(model.predict(valid_curves), axis=1)

    confusion_matrix = sklearn.metrics.confusion_matrix(valid_labels, predictions)
    figure_cm = plot_confusion_matrix(confusion_matrix, ["PC", "AFP", "NTP"])
    # figure_cm.show()

    cm_image = plot_to_image(figure_cm)

    with file_writer_cm.as_default():
        tf.summary.image("Confusion Matrix", cm_image, step=epoch)


train_files = tf.io.gfile.glob("./tfrecord/train-*")
train_dataset = tf.data.TFRecordDataset(filenames=train_files)
train_dataset = train_dataset.map(lambda x: tf.py_function(read_tfrecord, [x], (tf.float32, tf.int8)))
train_dataset = train_dataset.batch(32)

valid_files = tf.io.gfile.glob("./tfrecord/val-*")
valid_dataset = tf.data.TFRecordDataset(filenames=valid_files)
valid_dataset = valid_dataset.map(lambda x: tf.py_function(read_tfrecord, [x], (tf.float32, tf.int8)))
valid_dataset = valid_dataset.batch(32)

# for parsed_record in valid_dataset.take(1):
#     local_view = parsed_record[0][0].numpy()
#     plt.plot(local_view, ".")
#     plt.show()

log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
file_writer_cm = tf.summary.create_file_writer(log_dir + "/cm")

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

cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

model.fit(train_dataset,
          epochs=3,
          validation_data=valid_dataset,
          callbacks=[tensorboard_callback, cm_callback],
          )


