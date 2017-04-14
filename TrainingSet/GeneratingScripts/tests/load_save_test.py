import unittest

from filters_names import *
from load_save_script import FilteredImage
from load_save_script import load_from_file
from load_save_script import save_to_file

class LoadSaveTest(unittest.TestCase):
    def test_1(self):
        urls = ["url1", "url2", "url3"]
        applied_filters = [
            [Filters.NEGATIVE],
            [Filters.NEGATIVE, Filters.GRAYSCALE],
            [Filters.NEGATIVE, Filters.GRAYSCALE, Filters.GAUSSIAN_BLUR]
        ]
        training_set = [FilteredImage(urls[i], applied_filters[i]) for i in range(len(urls))]

        file_name = "test1"
        save_to_file(file_name, training_set)
        loaded_set = load_from_file(file_name)

        for i in range(len(loaded_set)):
            original = training_set[i]
            loaded = loaded_set[i]

            self.assertEqual(original.URL, loaded.URL, "Wrong URLs")
            self.assertEqual(original.filters, loaded.filters, "Wrong applied filtres")


if __name__ == '__main__':
    unittest.main()