import os
from os.path import join
from IPython.display import Image, display
import numpy as np
from tensorflow.contrib.keras.api.keras.applications.resnet50 import preprocess_input
from tensorflow.contrib.keras.api.keras.applications import ResNet50
from tensorflow.contrib.keras.api.keras.applications.resnet50 import decode_predictions
from tensorflow.contrib.keras.api.keras.preprocessing.image import load_img, img_to_array

hot_dog_image_directory = './hot-dog-not-hot-dog/train/hot_dog'
pre_trained_data_path = './resnet50_weights_tf_dim_ordering_tf_kernels.h5'
hot_dog_paths = [join(hot_dog_image_directory, filename) for filename in ['2417.jpg', '3690.jpg']]


not_hot_dog_image_directory = './hot-dog-not-hot-dog/train/not_hot_dog'
not_hot_dog_paths = [join(not_hot_dog_image_directory, filename) for filename in ['197.jpg', '1164.jpg']]

img_paths = hot_dog_paths+not_hot_dog_paths

img_size=224

# function to read and prepare image for modelling

def read_and_prepare_images(img_paths, img_height=img_size, img_width=img_size):
    # image loading via load_image
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    # for storing image in 3-D tensors: img_to_array
    img_array = np.array([img_to_array(img) for img in imgs])
    # preprocess_input applies arithmetic on pixels so that their value is b/w -1 and 1
    output = preprocess_input(img_array)
    return output

# Creating model with pre-trained weights
custom_model = ResNet50(weights=pre_trained_data_path)
test_data = read_and_prepare_images(img_paths)
preds = custom_model.predict(test_data)

# decode_predictions extracts highest probablity for each image
most_likely = decode_predictions(preds, top=3)

for i, img_path in enumerate(img_paths):
    display(Image(img_path))
    print(most_likely[i])
