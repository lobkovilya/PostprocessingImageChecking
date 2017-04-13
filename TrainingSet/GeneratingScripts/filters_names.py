from enum import Enum

class Filters(Enum):
    GAUSSIAN_BLUR = 1
    GRAYSCALE = 2
    NEGATIVE = 3

    @classmethod
    def from_string(cls, name):
        return getattr(cls, name, None)

    def to_string(self):
        return self._name_
