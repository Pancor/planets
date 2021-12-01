import tensorflow as tf


def get_local_view_model(show_summary=False):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Reshape(target_shape=(201, 1), input_shape=(201,)))
    model.add(tf.keras.layers.Conv1D(8, 3, activation=tf.nn.relu))
    model.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))

    if show_summary:
        model.summary()

    return model
