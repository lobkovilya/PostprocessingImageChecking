from PIL import ImageFilter, Image
from filters_names import Filters
from TrainingSet.GeneratingScripts.main.python.libs.instagram_filters.filters.lomo import Lomo
import itertools
from os import path

IMAGE_SIZE = 300, 300

class FilterApplier:
    # filters = [ImageFilter.GaussianBlur(3)]
    filters = [ImageFilter.SMOOTH_MORE]
    # filters = (ImageFilter.BLUR, ImageFilter.DETAIL, ImageFilter.SMOOTH, ImageFilter.SHARPEN)

    def apply(self, image):
        for filter_count in range(1, len(self.filters) + 1):
            for filters_to_apply in itertools.combinations(self.filters, filter_count):
                modified_image = image
                for filter_to_apply in filters_to_apply:
                    modified_image = modified_image.filter(filter_to_apply)
                applied_filter_names = [Filters.from_string(filter_to_apply.name.upper()) for filter_to_apply in filters_to_apply]
                yield (modified_image, applied_filter_names)

    def apply_single_filter(self, image):
        for filter_to_apply in self.filters:
            modified_image = image.filter(filter_to_apply)
            yield (modified_image, [Filters.from_string(filter_to_apply.name.upper())])
        yield (image, [Filters.NO_FILTER])

    def apply_instagram_filter(self, image_path):
        modified_image = Lomo(image_path, './tmp.jpg')
        modified_image.apply()
        yield (modified_image.image().resize(IMAGE_SIZE), [Filters.LOMO])
        yield (Image.open(image_path).resize(IMAGE_SIZE), [Filters.NO_FILTER])
