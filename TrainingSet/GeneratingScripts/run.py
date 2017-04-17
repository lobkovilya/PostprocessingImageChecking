from upload_images_to_cloud import upload_images_to_cloud
from load_save_script import save_to_file, load_from_file
from filter_applier import FilterApplier
import itertools

urls_file = "urls/fishes_urls.txt"
modified_urls_file = "urls/modified_fishes_urls.txt"
applier = FilterApplier()
image_descriptors = load_from_file(urls_file)
for desc in itertools.islice(image_descriptors, 0, 3):
    image = desc.get_image()
    modified_image_descriptors = upload_images_to_cloud(applier.apply(image))
    save_to_file(modified_urls_file, modified_image_descriptors)
