# setup
from fileinput import filename
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils
from PIL import Image
import os


def PrepareImage(file):
    img_path = os.path.join(os.path.dirname(__file__),
                            '../small_flower_dataset/')

    # im = Image.open(img_path)
    # im.show()

    # MobileNet expects 224x224 image sizes
    img = image.load_img(img_path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    # Processes images by scaling image RGB values
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)

    return tf.keras.applications.mobilenet_v2.preprocess_input(img_array_expanded_dims)


if __name__ == "__main__":
    print("Flower Classifier by Jarryd Stringfellow and Chase Dart")

    # Retrieve MobileNetV2 Data
    mobile = tf.keras.applications.mobilenet_v2.MobileNetV2()

    file = 'daisy/162362896_99c7d851c8_n.jpg'

    preprocessed_image = PrepareImage(file)

    predictions = mobile.predict(preprocessed_image)
    # Returns top 5 predictions
    results = imagenet_utils.decode_predictions(predictions)

    print(results)
