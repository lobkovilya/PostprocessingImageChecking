from PIL import ImageFilter
from filters_names import Filters
import itertools


class FilterApplier:
    filters = [ImageFilter.GaussianBlur(3)]
    # filters = [ImageFilter.BLUR]
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
