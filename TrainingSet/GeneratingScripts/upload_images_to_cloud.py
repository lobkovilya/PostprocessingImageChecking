from image_uploader import *


def upload_images_to_cloud(images_with_filter_names):
    uploader = ImageUploader()
    uploaded_images_descriptors = []
    for image_with_filter_names in images_with_filter_names:
        url = uploader.upload_image(image_with_filter_names[0])
        uploaded_images_descriptors.append(FilteredImage(url, image_with_filter_names[1]))
        print(url)
    return uploaded_images_descriptors
