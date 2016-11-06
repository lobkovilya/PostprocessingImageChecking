from PIL import Image, ImageChops
import exifread
import io

def show_diff_with_thumbnail(filename):
    f = open(filename, 'rb')
    tags = exifread.process_file(f)
    thumbnail_is_found = False
    thumbnail_valid_area_is_found = False

    for tag in tags.keys():
        if tag in 'JPEGThumbnail':
            thumbnail_is_found = True
            thumb_img = Image.open(io.BytesIO(tags[tag]))
        if tag in 'MakerNote ThumbnailImageValidArea':
            thumbnail_valid_area_is_found = True
            width_start = int(tags[tag].values[0])
            width_end = int(tags[tag].values[1] + 1)
            height_start = int(tags[tag].values[2])
            height_end = int(tags[tag].values[3] + 1)

    if thumbnail_is_found:
        if thumbnail_valid_area_is_found:
            thumb_img = thumb_img.crop((width_start, height_start, width_end, height_end))

        original_img = Image.open(filename)
        thumb_size = thumb_img.size[0], thumb_img.size[1]
        original_img.thumbnail(thumb_size, Image.BILINEAR)
        diff_img = ImageChops.difference(thumb_img, original_img)
        diff_img.show()

        diff_img.save("DiffBetweenThumbnails.bmp")
        thumb_img.save("ThumbnailFromMetadata.bmp")
        original_img.save("ThumbnailThatWeMade.bmp")
    else:
        print("Can't find a thumbnail in metadata.")

show_diff_with_thumbnail("3.jpg")