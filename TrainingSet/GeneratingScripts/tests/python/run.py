import itertools
import unittest

from filters_names import *
from load_save_script import FilteredImage
from load_save_script import load_from_file
from load_save_script import append_to_file
from upload_images_to_cloud import upload_images_to_cloud
from filter_applier import FilterApplier


class FilterApplierTest(unittest.TestCase):
    def test_1(self):
        urls_file = "../resources/urls/fishes_urls.txt"
        modified_urls_file = "../resources/urls/modified_fishes_urls.txt"
        applier = FilterApplier()
        image_descriptors = load_from_file(urls_file)
        for desc in itertools.islice(image_descriptors, 0, 1):
            image = desc.get_image()
            modified_image_descriptors = upload_images_to_cloud(applier.apply(image))
            append_to_file(modified_urls_file, modified_image_descriptors)

if __name__ == '__main__':
    unittest.main()