from sklearn.metrics import confusion_matrix

from plot import plot_confusion_matrix, plot_images


def eval_confusion_matrix(cls_true, cls_pred, num_classes):
    cm = confusion_matrix(y_true=cls_true, y_pred=cls_pred)
    print(cm)
    plot_confusion_matrix(cm, num_classes)


def eval_errors(data, image_shape, correct, cls_pred):
    incorrect = (correct == False)
    images = data.test.images[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = data.test.cls[incorrect]
    plot_images(images[0:9], image_shape, cls_true[0:9], cls_pred[0:9])

