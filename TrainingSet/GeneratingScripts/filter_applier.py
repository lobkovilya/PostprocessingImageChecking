from PIL import ImageFilter
from filters_names import Filters
import itertools


class FilterApplier:
    filters = (ImageFilter.BLUR, ImageFilter.DETAIL, ImageFilter.SMOOTH, ImageFilter.SHARPEN)

    def apply(self, image):
        for filter_count in range(1, len(self.filters) + 1):
            for filters_to_apply in itertools.combinations(self.filters, filter_count):
                modified_image = image
                for filter in filters_to_apply:
                    modified_image = modified_image.filter(filter)
                applied_filter_names = [Filters.from_string(filter.name.upper()) for filter in filters_to_apply]
                yield (modified_image, applied_filter_names)
