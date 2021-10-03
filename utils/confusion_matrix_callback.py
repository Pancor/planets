import matplotlib.pyplot as plt
import numpy as np
import itertools
import tensorflow as tf
import sklearn.metrics
import io


class ConfusionMatrixCallback(tf.keras.callbacks.Callback):

    def __init__(self, log_dir, dataset):
        super(ConfusionMatrixCallback, self).__init__()
        self.file_writer_cm = tf.summary.create_file_writer(log_dir + "/cm")

        (curves, labels) = self.__extract_dataset(dataset)
        self.curves = curves
        self.labels = labels

    def on_epoch_end(self, epoch, logs=None):
        self.__log_confusion_matrix(epoch)

    @staticmethod
    def __extract_dataset(dataset):
        curves = []
        labels = []

        for batch in dataset:
            for curve in batch[0]:
                curves.append(curve.numpy())
            for label in batch[1]:
                labels.append(label.numpy())

        curves = np.array(curves)
        labels = np.array(labels)
        return curves, labels

    def __log_confusion_matrix(self, epoch):
        predictions = np.argmax(self.model.predict(self.curves), axis=1)
        confusion_matrix = sklearn.metrics.confusion_matrix(self.labels, predictions)

        figure_cm = self.__plot_confusion_matrix(confusion_matrix, ["PC", "AFP", "NTP"])
        # figure_cm.show()
        cm_image = self.__plot_to_image(figure_cm)

        with self.file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)

    @staticmethod
    def __plot_confusion_matrix(cm, class_names):
        figure = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap("Blues"))
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
        threshold = cm.max() / 2.

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return figure

    @staticmethod
    def __plot_to_image(figure):
        buf = io.BytesIO()
        plt.savefig(buf, format='png')

        plt.close(figure)
        buf.seek(0)

        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        return image