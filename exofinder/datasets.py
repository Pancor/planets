import tensorflow as tf


def convert_test_dataset_record(local_curves, global_curves):
    return {
               "local_flux_input": local_curves[0],
               "global_flux_input": global_curves
           }, local_curves[1], local_curves[2]


def convert_valid_dataset_record(local_curves, global_curves):
    return {
               "local_flux_input": local_curves[0],
               "global_flux_input": global_curves
           }, local_curves[1]


def set_shape_for_weight(inputs, labels, sample_weights):
    return inputs, labels, tf.reshape(sample_weights, (len(sample_weights),))


def get_flux_series(path_to_files, flux_type="local"):
    files = tf.io.gfile.glob(path_to_files)
    dataset = tf.data.TFRecordDataset(filenames=files)

    if flux_type == "local":
        return dataset.map(lambda x: tf.py_function(__read_local_tfrecord, [x], (tf.float32, tf.int8)))
    elif flux_type == "global":
        return dataset.map(lambda x: tf.py_function(__read_global_tfrecord, [x], tf.float32))
    else:
        error_msg = "Provided flux_type : " + flux_type + " is not supported."
        raise ValueError(error_msg)


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


def __read_global_tfrecord(record):
    features = {
        "global_view": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
    }

    example = tf.io.parse_single_example(record, features)
    global_light_curve = example["global_view"]

    return global_light_curve
