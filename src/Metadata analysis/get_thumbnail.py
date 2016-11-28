from PIL import Image
import exifread
import io

def get_thumbnail(image):
    tags = exifread.process_file(image)

    for tag in tags.keys():
        if tag in 'JPEGThumbnail':
            thumbnail = Image.open(io.BytesIO(tags[tag]))
            thumbnail.save("thumbnail1.jpg")
            thumbnail.show()
            break

filename = "images\\1.jpg"
image = open(filename, 'rb')
get_thumbnail(image)
