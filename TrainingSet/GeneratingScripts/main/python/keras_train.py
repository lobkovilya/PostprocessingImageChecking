import os
import numpy as np

from pathlib import Path
from PIL import Image
from keras import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from matplotlib import pyplot as plt
from load_save_script import load_from_file

FILE_NAME = "dataset_from_site_photos"
# FILE_NAME = "dataset"
CACHE_DIRECTORY = "./cache_directory"
IMAGE_SIZE = 100, 100
CHANNELS = 3


def load_image(filtered_image):
    file_name = filtered_image.id + '_img.jpeg'
    file_path = os.path.join(CACHE_DIRECTORY, file_name)
    file = Path(file_path)

    if not file.exists():
        img = filtered_image.get_image()
        img.save(file_path)

    img = Image.open(file)
    img = img.resize(IMAGE_SIZE)
    return np.asarray(img, dtype="int32").ravel()


if __name__ == '__main__':
    image_set = load_from_file(FILE_NAME)

    print('Set loaded')
    images = []
    labels = []

    j = 0
    for filtered_image in image_set:
        image = load_image(filtered_image)
        print(str(j))
        j += 1
        images.append(image)
        labels.append([1 if x is True else 0 for x in filtered_image.get_mask()])
        # labels.append(filtered_image.get_mask().index(True))

    print(len(labels))

    images = np.vstack(images)
    labels = np.vstack(labels)

    test_image = images[1500:, ...]
    test_labels = labels[1500:, ...]

    images = images[:1500, ...]
    labels = labels[:1500, ...]

    # print(images[1:2, ...].shape)

    model = Sequential()

    model.add(Dense(64, input_dim=len(images[0]), init='uniform'))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.3))
    model.add(Dense(32, init='uniform'))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.3))
    model.add(Dense(4, init='uniform'))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.1, momentum=0.0, decay=0.0)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    history = model.fit(images, labels,
                        nb_epoch=20,
                        batch_size=100,
                        verbose=1,
                        validation_data=(test_image, test_labels))

    score = model.evaluate(test_image, test_labels, batch_size=60)
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
