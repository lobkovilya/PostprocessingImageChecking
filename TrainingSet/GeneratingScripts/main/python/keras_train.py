import os
import numpy as np

from pathlib import Path
from PIL import Image
from keras import Sequential, Input, Model
from keras.layers import Dense, Activation, Dropout, Convolution2D, MaxPooling2D, Flatten, Conv2D
from keras.optimizers import SGD
from keras.optimizers import rmsprop
from matplotlib import pyplot as plt
import matplotlib.cm as cm
# from keras.utils import plot_model
from keras.utils.vis_utils import plot_model
import numpy.ma as ma
from load_save_script import load_from_file
from mpl_toolkits.axes_grid1 import make_axes_locatable

# FILE_NAME = "dataset_single_3"
# FILE_NAME = "dataset_from_site_photos"
# FILE_NAME = "dataset_single_sharpen"
# FILE_NAME = "dataset"
# FILE_NAME = "dataset_instagram_lomo"
FILE_NAME = "dataset_single_smooth_more"
CACHE_DIRECTORY = "./cache_directory"
IMAGE_SIZE = 100, 100
CHANNELS = 3


def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
    """Wrapper around pl.imshow"""
    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    plt.colorbar(im, cax=cax)


def load_image(filtered_image):
    file_name = filtered_image.id + '_img.jpeg'
    file_path = os.path.join(CACHE_DIRECTORY, file_name)
    file = Path(file_path)

    if not file.exists():
        img = filtered_image.get_image()
        img.save(file_path)

    img = Image.open(file)
    img = img.resize(IMAGE_SIZE)
    return np.asarray(img, dtype="int32")


def make_mosaic(imgs, nrows, ncols, border=1):
    """
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
    nimgs = imgs.shape[0]
    imshape = imgs.shape[1:]

    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border),
                           dtype=np.float32)

    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in xrange(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols

        mosaic[row * paddedh:row * paddedh + imshape[0],
        col * paddedw:col * paddedw + imshape[1]] = imgs[i]
    return mosaic


if __name__ == '__main__':
    image_set = load_from_file(FILE_NAME)

    print('Set loaded')
    images = []
    labels = []

    j = 0
    for filtered_image in image_set:
        image = load_image(filtered_image)
        print(j)
        j += 1
        images.append(image)
        labels.append([1 if x is True else 0 for x in filtered_image.get_mask()])
        # labels.append(filtered_image.get_mask().index(True))

    print(labels)

    images = np.stack(images)
    labels = np.stack(labels)

    train_num = 900
    print(len(images))

    test_images = images[train_num:, ...]
    test_labels = labels[train_num:, ...]

    images = images[:train_num, ...]
    labels = labels[:train_num, ...]

    print(len(images))

    # model = Sequential()
    #
    # model.add(Dense(64, input_dim=4, init='uniform'))
    # model.add(Activation('sigmoid'))
    # model.add(Dropout(0.3))
    # model.add(Dense(32, init='uniform'))
    # model.add(Activation('sigmoid'))
    # model.add(Dropout(0.3))
    # model.add(Dense(4, init='uniform'))
    # model.add(Activation('softmax'))

    batch_size = 32  # in each iteration, we consider 32 training examples at once
    num_epochs = 10  # we iterate 200 times over the entire training set
    kernel_size = 3  # we will use 3x3 kernels throughout
    pool_size = 2  # we will use 2x2 pooling throughout
    conv_depth_1 = 32  # we will initially have 32 kernels per conv. layer...
    conv_depth_2 = 64  # ...switching to 64 after the first pooling layer
    drop_prob_1 = 0.25  # dropout after pooling with probability 0.25
    drop_prob_2 = 0.5  # dropout in the FC layer with probability 0.5
    hidden_size = 512  # the FC layer will have 512 neurons

    height = 100
    width = 100
    depth = 3

    num_classes = 2

    # inp = Input(shape=(height, width, depth))  # depth goes last in TensorFlow back-end (first in Theano)
    # # Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer)
    # conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
    # conv_2 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(conv_1)
    # pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
    # # drop_1 = Dropout(drop_prob_1)(pool_1)  можно убрать
    # # Conv [64] -> Conv [64] -> Pool (with dropout on the pooling layer)
    # conv_3 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(pool_1)
    # conv_4 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(conv_3)
    # pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)
    # # drop_2 = Dropout(drop_prob_1)(pool_2)
    # # Now flatten to 1D, apply FC -> ReLU (with dropout) -> softmax
    # flat = Flatten()(pool_2)
    # hidden = Dense(hidden_size, activation='relu')(flat)
    # drop_3 = Dropout(drop_prob_2)(hidden)
    # out = Dense(num_classes, activation='softmax')(drop_3)
    #
    # model = Model(inputs=inp, outputs=out)  # To define a model, just specify its input and output layers
    # # sgd = SGD(lr=0.1, momentum=0.0, decay=0.0)
    # sgd = SGD(lr=0.01, decay=1e-6)
    # model.compile(loss='categorical_crossentropy',  # using the cross-entropy loss function
    #               optimizer='adam',  # using the Adam optimiser
    #               metrics=['accuracy'])  # reporting the accuracy

    # _________________
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=images.shape[1:],
                     kernel_initializer='random_uniform', ))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    opt = rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    # _________________

    history = model.fit(images, labels,  # Train the model using the training set...
              batch_size=batch_size, epochs=num_epochs,
              verbose=1, validation_split=0.1, shuffle=True)  # ...holding out 10% of the data for validation
    score = model.evaluate(test_images, test_labels, verbose=1)  # Evaluate the trained model on the test set!

    # W = conv_1.get_value(borrow=True)
    # W = np.squeeze(W)
    # print("W shape : ", W.shape)
    #
    # plt.figure(figsize=(15, 15))
    # plt.title('conv1 weights')
    # nice_imshow(plt.gca(), make_mosaic(W, 6, 6), cmap=cm.binary)

    # sgd = SGD(lr=0.1, momentum=0.0, decay=0.0)
    # model.compile(loss='categorical_crossentropy',
    #               optimizer=sgd,
    #               metrics=['accuracy'])
    #
    # history = model.fit(images, labels,
    #                     nb_epoch=20,
    #                     batch_size=100,
    #                     verbose=1,
    #                     validation_data=(test_images, test_labels))

    # score = model.evaluate(test_images, test_labels, batch_size=60)
    print(score)

    train_error = 1 - np.array(history.history['acc'])
    val_error = 1 - np.array(history.history['val_acc'])

    plt.figure(figsize=(12, 8))
    plt.plot(train_error, label='train')
    plt.plot(val_error, label='test')
    plt.title('fMNIST error')
    plt.xlabel('Train epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.show()
