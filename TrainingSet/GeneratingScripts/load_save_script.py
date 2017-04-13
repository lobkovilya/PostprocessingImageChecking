import json
from io import BytesIO
from json import JSONEncoder, JSONDecoder

import requests
from PIL import Image

from filters_names import *


class FilteredImage:
    def __init__(self, imageURL, filters):
        self.URL = imageURL
        self.filters = filters

    def get_image(self):
        response = requests.get(self.URL)
        return Image.open(BytesIO(response.content))

    def get_mask(self):
        return FilterMask(self.filters).get_mask()


class FilterMask:
    def __init__(self, applied_filters):
        self.filters = {f: False for f in Filters}
        for filter in applied_filters:
            self.filters[filter] = True

    def get_mask(self):
        return [v for v in self.filters.values()]


class TrainingSetEncoder(JSONEncoder):
    def default(self, data):
        if isinstance(data, FilteredImage):
            return {"URL": data.URL, "Filters": data.filters}
        if isinstance(data, Filters):
            return {"FilterName": data.to_string()}

        return super().default(data)


class TrainingSetDecoder(JSONDecoder):
    def __init__(self):
        super().__init__(object_hook=self.object_hook)

    def object_hook(self, d):
        if "Filters" in d:
            url = d["URL"]
            filters = d["Filters"]
            return FilteredImage(url, filters)
        if "FilterName" in d:
            filter_name = Filters.from_string(d["FilterName"])
            return filter_name

        return d

def save_to_file(file, data):
    with open(file, "w") as f:
        json.dump(data, f, cls=TrainingSetEncoder, indent=3)


def load_from_file(file):
    with open(file, "rb") as f:
        return json.load(f, cls=TrainingSetDecoder)


images = [FilteredImage("http://www.pet4me.ru/sites/default/files/imagecache/1000x667/5748_original.jpg",
                        [Filters.GAUSSIAN_BLUR, Filters.NEGATIVE]),
          FilteredImage("http://gotonight.ru/ih800/catalog/places2/708_kotiki11.png",
                        [Filters.NEGATIVE, Filters.GRAYSCALE]),
          FilteredImage("http://cdn.fishki.net/upload/post/201406/23/1279678/xYjUKHJssWQ.jpg", [Filters.GAUSSIAN_BLUR])]

file_name = "training_set"
# save_to_file(file_name, images)
images_from_file = load_from_file(file_name)

for i in images_from_file:
    # print(i)
    i.get_image().show(i.URL)
