import cv2
import json
import numpy as np
from sklearn import metrics
import itertools
from sklearn.svm import SVC
from feature_generator import image_features
from load_save_script import load_from_file


X = []
y = []
urls_file = "../resources/urls/fishes_urls.txt"
modified_urls_file = "../resources/urls/modified_fishes_urls.txt"

image_descriptors = load_from_file(urls_file)
for desc in itertools.islice(image_descriptors, 0,7):
    image = desc.get_image()
    image.save("tmp.jpg")
    rgb_image = cv2.imread("tmp.jpg", cv2.IMREAD_COLOR)
    features = image_features(rgb_image, (3, 2))
    y.append("correct")
    X.append(features)

modified_image_descriptors = load_from_file(modified_urls_file)
for desc in itertools.islice(modified_image_descriptors, 0,7):
    image = desc.get_image()
    image.save("tmp.jpg")
    rgb_image = cv2.imread("tmp.jpg", cv2.IMREAD_COLOR)
    features = image_features(rgb_image, (3, 2))
    y.append("corrupted")
    X.append(features)

X = np.array(X)
y = np.array(y)
clf = SVC()
clf.fit(X, y)
params = clf.get_params()
with open("svm_params.json", "w") as f:
    json.dump(params, f)

X_test = []
y_test = []

for desc in itertools.islice(image_descriptors, 7, 15):
       image = desc.get_image()
       image.save("tmp.jpg")
       rgb_image = cv2.imread("tmp.jpg", cv2.IMREAD_COLOR)
       features = image_features(rgb_image, (3, 2))
       y_test.append("correct")
       X_test.append(features)

modified_image_descriptors = load_from_file(modified_urls_file)
for desc in itertools.islice(modified_image_descriptors, 7, 15):
        image = desc.get_image()
        image.save("tmp.jpg")
        rgb_image = cv2.imread("tmp.jpg", cv2.IMREAD_COLOR)
        features = image_features(rgb_image, (3, 2))
        y_test.append("corrupted")
        X_test.append(features)

X = np.array(X)
y = np.array(y)
predicted = clf.predict(X_test)
print(metrics.classification_report(y_test, predicted))
print(metrics.confusion_matrix(y_test, predicted))