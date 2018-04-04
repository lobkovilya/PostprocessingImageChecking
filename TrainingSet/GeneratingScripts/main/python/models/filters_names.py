from enum import Enum

class Filters(Enum):
    NO_FILTER = 0
    SMOOTH_MORE = 1
    # LOMO = 1
    # BLUR = 1
    # DETAIL = 2
    # SMOOTH = 3
    # SHARPEN = 1
    # SMOOTH_MORE = 2

    @classmethod
    def from_string(cls, name):
        return getattr(cls, name, None)

    def to_string(self):
        return self._name_
