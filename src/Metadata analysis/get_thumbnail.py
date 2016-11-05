from PIL import Image
import exifread
import io

def get_thumbnail(filename):
    f = open(filename, 'rb')
    tags = exifread.process_file(f)

    for tag in tags.keys():
        if tag in 'JPEGThumbnail':
            image = Image.open(io.BytesIO(tags[tag]))
            image.save("thumbnail1.jpg")
            image.show()
            break

get_thumbnail("1.jpg")