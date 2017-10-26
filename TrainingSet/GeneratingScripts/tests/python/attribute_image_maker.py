import cv2


class ImageAttributes:
    def __init__(self, rgb_image):
        self.__calculate_mean_rgb_values(rgb_image)
        self.__calculate_min_max_rgb_values(rgb_image)

        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        self.__calculate_mean_grayscale_value(gray_image)
        self.__calculate_min_max_grayscale_value(gray_image)

    def __calculate_mean_grayscale_value(self, gray_image):
        self.mean_grayscale = cv2.mean(gray_image)[0]

    def __calculate_min_max_grayscale_value(self, gray_image):
        self.min_graycale, self.max_grayscale, none, none = cv2.minMaxLoc(gray_image)

    def __calculate_mean_rgb_values(self, color_image):
        self.mean_blue, self.mean_green, self.mean_red, none = cv2.mean(color_image)

    def __calculate_min_max_rgb_values(self, color_image):
        blue_channel, green_channel, red_channel = cv2.split(color_image)
        self.min_blue, self.max_blue, none, none = cv2.minMaxLoc(blue_channel)
        self.min_green, self.max_green, none, none = cv2.minMaxLoc(green_channel)
        self.min_red, self.max_red, none, none = cv2.minMaxLoc(red_channel)

    def get_attributes(self):
        return [self.mean_grayscale,
                self.min_graycale,
                self.max_grayscale,
                self.mean_red,
                self.mean_green,
                self.mean_blue,
                self.min_red,
                self.min_green,
                self.min_blue,
                self.max_red,
                self.max_green,
                self.max_blue]


# rgb_image = cv2.imread('img.jpg', cv2.IMREAD_COLOR)
# image_attributes = ImageAttributes(rgb_image)
#
# print(image_attributes.mean_grayscale)
# print(image_attributes.min_graycale)
# print(image_attributes.max_grayscale)
#
# print(image_attributes.mean_red)
# print(image_attributes.mean_green)
# print(image_attributes.mean_blue)
#
# print(image_attributes.min_red)
# print(image_attributes.min_green)
# print(image_attributes.min_blue)
#
# print(image_attributes.max_red)
# print(image_attributes.max_green)
# print(image_attributes.max_blue)
