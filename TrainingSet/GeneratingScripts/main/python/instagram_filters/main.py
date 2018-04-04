import matplotlib.pyplot as plt
import skimage
from skimage import io
from skimage import filters
from TrainingSet.GeneratingScripts.main.python.libs.instagram_filters.filters.gotham import Gotham
from TrainingSet.GeneratingScripts.main.python.libs.instagram_filters.filters.toaster import Toaster
from TrainingSet.GeneratingScripts.main.python.libs.instagram_filters.filters.kelvin import Kelvin
from TrainingSet.GeneratingScripts.main.python.libs.instagram_filters.filters.lomo import Lomo
from TrainingSet.GeneratingScripts.main.python.libs.instagram_filters.filters.nashville import Nashville
import numpy as np

import matplotlib
matplotlib.rcParams['xtick.major.size'] = 0
matplotlib.rcParams['ytick.major.size'] = 0
matplotlib.rcParams['xtick.labelsize'] = 0
matplotlib.rcParams['ytick.labelsize'] = 0

IMAGE_PATH = 'Lenna.jpg'
# IMAGE_PATH = '../web_scraping/images/0b87daf9-0eab-4b5f-b435-3ee4c6c2a511.jpg'


def sharpen(image, a, b, sigma=10):
    blurred = filters.gaussian(image, sigma=sigma, multichannel=True)
    return np.clip(image * a - blurred * b, 0, 1.0)


def before_after_show(before, after):
    fig = plt.figure()
    before_subplot = fig.add_subplot(1, 2, 1)
    before_subplot.set_title('Before')
    plt.imshow(before)
    after_subplot = fig.add_subplot(1, 2, 2)
    after_subplot.set_title('After')
    plt.imshow(after)

    plt.show()


def channel_adjust(channel, values):
    orig_size = channel.shape
    flat_channel = channel.flatten()
    adjusted = np.interp(
        flat_channel,
        np.linspace(0, 1, len(values)),
        values)

    return adjusted.reshape(orig_size)

def split_image_into_channels(image):
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]
    return red_channel, green_channel, blue_channel


def merge_channels(red_channel, green_channel, blue_channel):
    return np.stack([red_channel, green_channel, blue_channel], axis=2)


def gotham(image):
    r = image[:, :, 0]
    b = image[:, :, 2]
    r_boost_lower = channel_adjust(r, [
        0, 0.05, 0.1, 0.2, 0.3,
        0.5, 0.7, 0.8, 0.9,
        0.95, 1.0])
    b_more = np.clip(b + 0.03, 0, 1.0)
    merged = np.stack([r_boost_lower, image[:, :, 1], b_more], axis=2)
    blurred = filters.gaussian(merged, sigma=10, multichannel=True)
    final = np.clip(merged * 1.3 - blurred * 0.3, 0, 1.0)
    b = final[:, :, 2]
    b_adjusted = channel_adjust(b, [
        0, 0.047, 0.118, 0.251, 0.318,
        0.392, 0.42, 0.439, 0.475,
        0.561, 0.58, 0.627, 0.671,
        0.733, 0.847, 0.925, 1])
    final[:, :, 2] = b_adjusted
    return final


if __name__ == '__main__':
    f = Kelvin(IMAGE_PATH, 'Lenna_kelvin.jpg')
    f.apply()

    f = Toaster(IMAGE_PATH, 'Lenna_toaster.jpg')
    f.apply()

    f = Lomo(IMAGE_PATH, 'Lenna_lomo.jpg')
    f.apply()

    f = Nashville(IMAGE_PATH, 'Lenna_nash.jpg')
    f.apply()

    f = Gotham(IMAGE_PATH, 'Lenna_gotham.jpg')
    f.apply()
    # original_image = skimage.img_as_float(io.imread(IMAGE_PATH))
    # sharper = gotham(original_image)
    #
    # before_after_show(original_image, sharper)
