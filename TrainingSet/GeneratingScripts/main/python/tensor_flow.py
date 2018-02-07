from pathlib import Path

import os
import tensorflow as tf
import numpy as np
from PIL import Image

from load_save_script import load_from_file


FILE_NAME = "dataset_single"
CACHE_DIRECTORY = "./cache_directory"
IMAGE_SIZE = [100, 100]
CHANNELS = 3


def load_image(filtered_image):
    file_name = filtered_image.id + '_img.jpeg'
    file_path = os.path.join(CACHE_DIRECTORY, file_name)
    file = Path(file_path)

    if file.exists():
        tensor = tf.image.decode_jpeg(Image.open(file).tobytes(), CHANNELS)
    else:
        img = filtered_image.get_image()
        img.save(file_path)

        tensor = tf.image.decode_jpeg(img.tobytes(), CHANNELS)

    return tf.image.resize_images(tensor, IMAGE_SIZE)


def get_train_inputs(batch_size, images, labels):

    def train_inputs():
        return {"x": images}, labels
        # dataset = tf.contrib.data.Dataset.from_tensor_slices((images, labels))
        # dataset = dataset.batch(batch_size)
        # dataset.
        # iterator = dataset.make_initializable_iterator()
        # next_example, next_label = iterator.get_next()
        # return next_example, next_label

    return train_inputs


if __name__ == '__main__':
    image_set = load_from_file(FILE_NAME)

    images = []
    labels = []

    for filtered_image in image_set:
        image = load_image(filtered_image)
        images.append(image)
        labels.append(filtered_image.get_mask().index(True))
        # labels.append([1 if x is not None else 0 for x in filtered_image.get_mask()])

    images = tf.stack(images)
    labels = tf.stack(labels)

    # print(images.shape, labels.shape)

    feature_columns = [tf.feature_column.numeric_column("x", shape=[100, 100, 3])]
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=4,
                                            model_dir="/tmp")

    # train_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={"x": np.array(images.eval())},
    #     y=np.array(labels.eval()),
    #     num_epochs=None,
    #     shuffle=True)

    # classifier.fit(input_fn=get_train_inputs(64, images, labels), steps=2000)
    # classifier.train(input_fn=train_input_fn, steps=2000)
    classifier.train(input_fn=get_train_inputs(64, images, labels), steps=2000)

    print("Done")
    # dataset = tf.contrib.data.Dataset.from_tensor_slices((images, labels))
    # dataset = dataset.batch(3)
    #
    # iterator = dataset.make_initializable_iterator()
    # next_example, next_label = iterator.get_next()




# if __name__ == '__main__':
#     from tensorflow.examples.tutorials.mnist import input_data
#
#     mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#     print(type(mnist.train.images))