from os import listdir
from os.path import isfile, join

from image_uploader import ImageUploader
from upload_images_to_cloud import upload_images_to_cloud
from filter_applier import FilterApplier
from load_save_script import append_to_file
from load_save_script import save_to_file
from PIL import Image

if __name__ == '__main__':
    uploader = ImageUploader()

    # im = Image.open("./cache_directory/4d30279a-0a2a-4e07-8204-7192105d1894_img.bmp")
    # url = uploader.upload_image_file("./images/panda.jpg")

    mypath = "C:\\Home\\NSU\\Diploma\\PostprocessingImageChecking\\TrainingSet\\GeneratingScripts\\main\\python\\web_scraping\\images"
    onlyfiles = [join(mypath,f) for f in listdir(mypath) if isfile(join(mypath, f))]
    applier = FilterApplier()

    i = 1;
    descriptors = []
    for f in onlyfiles:
        modified_image_descriptors = upload_images_to_cloud(applier.apply(Image.open(f)))
        descriptors += modified_image_descriptors
        # append_to_file("./dataset_from_photo_site", modified_image_descriptors)
        print(str(i) + "Upload " + f)
        i += 1
        save_to_file('./dataset_from_site_photos', descriptors)