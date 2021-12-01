import matplotlib.pyplot as plt
import numpy as np
import itertools
import tensorflow as tf
import sklearn.metrics
import io


class ConfusionMatrixCallback(tf.keras.callbacks.Callback):

    is_showing_cm = False

    def __init__(self, log_dir, dataset):
        super(ConfusionMatrixCallback, self).__init__()
        self.file_writer_cm = tf.summary.create_file_writer(log_dir + "/cm")

        (curves, labels) = self.__extract_dataset(dataset)
        self.curves = curves
        self.labels = labels

    def on_epoch_end(self, epoch, logs=None):
        self.__log_confusion_matrix(epoch)

    def set_showing_confusion_matrix(self, is_showing_cm):
        self.is_showing_cm = is_showing_cm

    @staticmethod
    def __extract_dataset(dataset):
        curves = []
        labels = []

        for data in dataset:
            curves.append(data[0].numpy())
            labels.append(data[1].numpy())

        curves = np.array(curves)
        labels = np.array(labels)
        return curves, labels

    def __log_confusion_matrix(self, epoch):
        raw_predictions = self.model.predict(self.curves)
        labeled_predictions = np.array([self.__convert_predicts_to_labels(pred) for pred in raw_predictions])
        confusion_matrix = sklearn.metrics.confusion_matrix(self.labels, labeled_predictions)

        figure_cm = self.__plot_confusion_matrix(confusion_matrix, ["PC", "NTP"])
        if self.is_showing_cm:
            figure_cm.show()
        cm_image = self.__plot_to_image(figure_cm)

        with self.file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)

    @staticmethod
    def __convert_predicts_to_labels(prediction):
        return int(round(prediction[0]))

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
