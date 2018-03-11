from io import BytesIO

import cloudinary
import cloudinary.api
import cloudinary.uploader
from load_save_script import *


class ImageUploader:
    def __init__(self):
        # cloudinary.config(
        #     cloud_name="div9zoukt",
        #     api_key="826937426924497",
        #     api_secret="WPz-7D9zwqq7uj-qBrXqpP19ZwY"
        # )
        cloudinary.config(
            cloud_name="dwiiw46pv",
            api_key="277186483556198",
            api_secret="QfDHkof17QIHLcxeLFZ2w7jgUxk"
        )

    def upload_image_file(self, file):
        return cloudinary.uploader.upload(file)["url"]

    def upload_image(self, image):
        raw_bytes = BytesIO()
        image.save(raw_bytes, "BMP")
        url = cloudinary.uploader.upload(raw_bytes.getvalue())["url"]
        raw_bytes.close()
        return url



