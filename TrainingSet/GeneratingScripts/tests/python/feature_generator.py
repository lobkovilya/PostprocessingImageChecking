from attribute_image_maker import ImageAttributes


def image_features(rgb_image, num_block):
    features = []
    width = rgb_image.shape[1]
    height = rgb_image.shape[0]
    block_width = width//num_block[0]
    block_height = height//num_block[1]

    for y in range(0, height - block_height + 1, block_height):
        for x in range(0, width - block_width + 1, block_width):
            img_block = rgb_image[y:y+block_height, x:x+block_width]
            attr = ImageAttributes(img_block)
            block_features = attr.get_attributes()
            features.extend(block_features)
    return features
