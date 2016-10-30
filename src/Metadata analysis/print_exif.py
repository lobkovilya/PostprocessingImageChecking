from PIL import Image
from PIL.ExifTags import TAGS

def print_exif(imagePath):
    image = Image.open(imagePath)
    exif_data = image._getexif()
    for tag, value in exif_data.items():
        print("%s: %s" % (TAGS[tag], value))

print_exif("1.jpg")