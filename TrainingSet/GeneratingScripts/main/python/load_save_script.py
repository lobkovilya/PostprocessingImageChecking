import json
import uuid
from io import BytesIO
from json import JSONEncoder, JSONDecoder

import requests
from PIL import Image

from filters_names import *


class FilteredImage:
    def __init__(self, imageURL, filters, id = None):
        self.URL = imageURL
        self.filters = filters
        self.id = str(uuid.uuid4()) if id is None else id

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
            return {"URL": data.URL, "Filters": data.filters, "Id": data.id}
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
            id = d["Id"]
            return FilteredImage(url, filters, id)
        if "FilterName" in d:
            filter_name = Filters.from_string(d["FilterName"])
            return filter_name

        return d


def save_to_file(file, data):
    with open(file, "w") as f:
        json.dump(data, f, cls=TrainingSetEncoder, indent=3)


def load_from_file(file):
    with open(file, "r") as f:
        return json.load(f, cls=TrainingSetDecoder)


def append_to_file(file, data):
    with open(file, "a") as f:
        json.dump(data, f, cls=TrainingSetEncoder, indent=3)


def to_csv(file, csv_file):
    size = 100, 100
    training_set = load_from_file(file)
    with open(csv_file, "w") as f:
        for filtered_img in training_set:
            img = filtered_img.get_image()
            img = img.resize(size, Image.ANTIALIAS)

            img_list = list(img.getdata())
            for pixel in img_list:
                f.write(str(pixel))
                f.write(', ')

            f.write(str(filtered_img.get_mask()))
            f.write('\n')


# img.save('./img', "JPEG")

if __name__ == '__main__':
    # to_csv('./three_images.txt', './csv_file');
    images_desc = load_from_file('fishes_urls_id.txt')
    for d in images_desc:
        print(d.id)

