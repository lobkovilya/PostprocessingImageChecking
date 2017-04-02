import pickle
import requests
import numpy as np
from io import BytesIO
from PIL import Image
from filters_names import *


class FilteredImage:
    def __init__(self, imageURL, filters):
        self.URL = imageURL
        self.filter_mask = FilterMask(filters)

    def get_image(self):
        response = requests.get(self.URL)
        return Image.open(BytesIO(response.content))

    def get_mask(self):
        return self.filter_mask.get_mask()


class FilterMask:
    def __init__(self, applied_filters):
        self.filters = {f: False for f in Filters}
        for filter in applied_filters:
            self.filters[filter] = True

    def get_mask(self):
        return [v for v in self.filters.values()]


def save_to_file(file, data):
    with open(file, "wb") as f:
        pickle.dump(data, f)


def load_from_file(file):
    with open(file, "rb") as f:
        return pickle.load(f)


# images = [FilteredImage("http://www.pet4me.ru/sites/default/files/imagecache/1000x667/5748_original.jpg", [Filters.GAUSSIAN_BLUR, Filters.NEGATIVE]),
#           FilteredImage("http://gotonight.ru/ih800/catalog/places2/708_kotiki11.png", [Filters.NEGATIVE, Filters.GRAYSCALE]),
#           FilteredImage("http://cdn.fishki.net/upload/post/201406/23/1279678/xYjUKHJssWQ.jpg", [Filters.GAUSSIAN_BLUR])]
#
# file_name = "training_set"
# save_to_file(file_name, images)
# images_from_file = load_from_file(file_name)
#
# for i in images_from_file:
#     i.get_image().show(i.URL)

