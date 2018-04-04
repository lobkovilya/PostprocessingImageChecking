import numpy as np

from keras.optimizers import rmsprop
from keras import Sequential
from keras.layers import Dense, Activation, Dropout, Convolution2D, MaxPooling2D, Flatten, Conv2D
from matplotlib import pyplot as plt
from load_save_script import load_from_file, load_image

FILE_NAME = "../dataset_single_smooth_more"
CACHE_DIRECTORY = "../cache_directory"
IMAGE_SIZE = 100, 100

batch_size = 32  # in each iteration, we consider 32 training examples at once
num_epochs = 10  # we iterate 200 times over the entire training set
kernel_size = 3  # we will use 3x3 kernels throughout
pool_size = 2  # we will use 2x2 pooling throughout
conv_depth_1 = 32  # we will initially have 32 kernels per conv. layer...
conv_depth_2 = 64  # ...switching to 64 after the first pooling layer
drop_prob_1 = 0.25  # dropout after pooling with probability 0.25
drop_prob_2 = 0.5  # dropout in the FC layer with probability 0.5
hidden_size = 512  # the FC layer will have 512 neurons

num_classes = 2

if __name__ == '__main__':
    image_set = load_from_file(FILE_NAME)

    print('Set loaded')
    images = []
    labels = []

    j = 0
    for filtered_image in image_set:
        image = load_image(filtered_image, CACHE_DIRECTORY, IMAGE_SIZE)
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
