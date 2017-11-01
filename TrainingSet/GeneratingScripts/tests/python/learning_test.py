import cv2
import json
import numpy as np
import itertools
from sklearn.svm import SVC
from feature_generator import image_features
from load_save_script import load_from_file
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit

X = []
y = []
urls_file = "../resources/urls/fishes_urls.txt"
modified_urls_file = "../resources/urls/modified_fishes_urls.txt"

image_descriptors = load_from_file(urls_file)  # 1307 urls
for desc in itertools.islice(image_descriptors, 0,500):
    image = desc.get_image()
    image.save("tmp.jpg")
    rgb_image = cv2.imread("tmp.jpg", cv2.IMREAD_COLOR)
    features = image_features(rgb_image, (3, 2))
    y.append("correct")
    X.append(features)

modified_image_descriptors = load_from_file(modified_urls_file)  # 975 urls
for desc in itertools.islice(modified_image_descriptors, 0,500):
    image = desc.get_image()
    image.save("tmp.jpg")
    rgb_image = cv2.imread("tmp.jpg", cv2.IMREAD_COLOR)
    features = image_features(rgb_image, (3, 2))
    y.append("corrupted")
    X.append(features)

X = np.array(X)
y = np.array(y)

C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(X, y)

print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))
with open("svm_params.json", "w") as f:
    json.dump(grid.best_params_, f)