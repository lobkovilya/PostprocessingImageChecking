from PIL import Image, ImageChops, ImageEnhance
import sys, os.path


def error_level_analysis(image):
    resaved = "./resaved.jpg"
    image.save(resaved, "JPEG", quality=95)
    resaved_im = Image.open(resaved)

    ela_im = ImageChops.difference(image, resaved_im)
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0/max_diff

    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)
    os.remove(resaved)
    return ela_im

filename = "/home/lobkov/NSU/Yandex/ImageCollection/2.jpg"
im = Image.open(filename)
error_level_analysis(im).show()


