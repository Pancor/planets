import tensorflow as tf


def load_local_view_with_labels(path):
    train_files = tf.io.gfile.glob(path)
    train_dataset = tf.data.TFRecordDataset(filenames=train_files)
    return __get_local_view_with_labels(train_dataset)


def __get_local_view_with_labels(dataset):
    return dataset.map(lambda x: tf.py_function(__read_local_tfrecord, [x], (tf.float32, tf.int8)))


def __read_local_tfrecord(record):
    features = {
        "local_view": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        "av_training_set": tf.io.FixedLenFeature([], tf.string, default_value=""),
    }

    example = tf.io.parse_single_example(record, features)
    local_light_curve = example["local_view"]
    av_training_set = example["av_training_set"]
    av_training_set = __convert_av_training_set(av_training_set)

    return local_light_curve, av_training_set


def __convert_av_training_set(av_training_set):
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
