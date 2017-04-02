import cloudinary
import cloudinary.uploader
import cloudinary.api
import os
import os.path

from load_save_script import *


class ImageUploader:
    def __init__(self):
        cloudinary.config(
            cloud_name="imagechecker",
            api_key="488922983197526",
            api_secret="559JXXeNYAY2MlKv9t6qHo3WIjs"
        )

    def upload_image(self, file):
        return cloudinary.uploader.upload(file)["url"]


uploader = ImageUploader()
folder = "./Images"
filtered_images = []

for f in os.listdir(folder):
    fname = folder + "/" + f
    if os.path.isfile(fname):
        print("Try to upload " + fname)
        url = uploader.upload_image(fname)
        filtered_images.append(FilteredImage(url, []))

cats = "cats"
save_to_file(cats, filtered_images)
from_file = load_from_file(cats)

for f in from_file:
    f.get_image().show()