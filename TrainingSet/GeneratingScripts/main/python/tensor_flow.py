from pathlib import Path

import os
import tensorflow as tf
from PIL import Image

from load_save_script import load_from_file

FILE_NAME = "three_images.txt"
CACHE_DIRECTORY = "./cache_directory"
IMAGE_SIZE = [100, 100]
CHANNELS = 3

def load_image(filtered_image):
    file_name = filtered_image.id + '_img.jpeg'
    file_path = os.path.join(CACHE_DIRECTORY, file_name)
    file = Path(file_path)

    if file.exists():
        tensor =  tf.image.decode_jpeg(Image.open(file).tobytes(), CHANNELS)
    else:
        img = filtered_image.get_image()
        img.save(file_path)

        tensor = tf.image.decode_jpeg(img.tobytes(), CHANNELS)

    return tf.image.resize_images(tensor, IMAGE_SIZE)


if __name__ == '__main__':
    image_set = load_from_file(FILE_NAME)

    tensors = []
    for filtered_image in image_set:
        tensor = load_image(filtered_image)
        tensors.append(tensor)

