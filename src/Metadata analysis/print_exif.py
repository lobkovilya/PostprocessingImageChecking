from PIL import Image
from PIL.ExifTags import TAGS

def print_exif(image):
    exif_data = image._getexif()
    for tag, value in exif_data.items():
        print("%s: %s" % (TAGS[tag], value))

image = Image.open("1.jpg")
print_exif(image)