import numpy as np
import matplotlib.pyplot as plt


def plot_data(data):
    print data.test.labels[0:5, :]
    data.test.cls = np.array([label.argmax() for label in data.test.labels])
    print data.test.cls[0:5]


def plot_images(images, image_shape, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
   
    figure, axes = plt.subplots(3, 3)
    figure.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(image_shape), cmap='binary')
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


def plot_confusion_matrix(cm, num_classes):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def plot_weights(weights, image_shape):
    w_min = np.min(weights)
    w_max = np.max(weights)

    figure, axes = plt.subplots(3, 4)
    figure.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        if i < 10:
            image = weights[:, i].reshape(image_shape)
            ax.set_xlabel("Weights: {0}".format(i))
            ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def plot_conv_weights(weights, input_channel=0):
    weight_min = np.min(weights)
    weight_max = np.max(weights)
    num_filters = weights.shape[3]

    num_grids = math.ceil(math.sqrt(num_filters))
    figure, axes = plt.subplots(num_grids, num_grids)

    for i, ax in enumerate(axes.flat):
        if (i < num_filters):
            image = weight[:, :, input_channel, i]
            ax.imshow(image, vmin=weight_min, vmax=weight_max, interpolation='nearest', cmap='seismic')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


def plot_conv_layer(image):
    num_filters = image.shape[3]
    image_data = image.data
    num_grids = math.ceil(math.sqrt(num_filters))
    figure, axes = plt.subplots(num_grids, num_grids)

    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            this_image = image_data[0, :, :, i]
            ax.imshow(this_image, interpolation='nearest', cmap='binary')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()
