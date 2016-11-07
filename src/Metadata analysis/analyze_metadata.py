from PIL import Image
import exifread

def analyze_metadata(filename):

    time1 = 0
    time2 = 0
    exif_height = 0
    exif_width = 0
    level_of_distrust = 0

    image = open(filename, 'rb')
    img = Image.open(filename)

    tags = exifread.process_file(image)
    for tag in tags.keys():
        print ("%s: %s" % (tag, tags[tag]))
        if tag in 'Image Software':
            level_of_distrust += 1
        if tag in 'Image DateTime':
            time1 = str(tags[tag])
        if tag in 'EXIF DateTimeDigitized':
            time2 = str(tags[tag])
        if tag in 'EXIF ExifImageWidth:':
            exif_width = int(tags[tag].values[0])
        if tag in 'EXIF ExifImageLength:':
            exif_height = int(tags[tag].values[0])

    if time1 and time2 and time1 != time2:
        level_of_distrust += 1
    if exif_width and exif_height and (img.size[0] != exif_width or img.size[1] != exif_height):
        level_of_distrust += 1

    print ("\nLevel of distrust of metadata (from 0 to 3): %d" % level_of_distrust)

analyze_metadata("insertion.jpg")