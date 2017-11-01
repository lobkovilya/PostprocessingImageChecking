import random


def corrupt_image(image, block_size):
    width = image.col
    height = image.row
    bx = random.randint(0, width // block_size)
    by = random.randint(0, height // block_size)
    x = bx * block_size
    y = by * block_size
    corrupted_blocks_numbers = [by * (width // block_size) + bx]
    corrupted_image = image

    # ...

    return corrupted_image, corrupted_blocks_numbers
