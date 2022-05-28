# setup
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils
import os

# Define dataset location
img_path = os.path.join(os.path.dirname(__file__),
                        '../small_flower_dataset/')


def PrepareImage(file):
    """
    Takes the individual image and prepares it by resizing and RGB scaling it for keras
    """
    # MobileNet expects 224x224 image sizes
    img = image.load_img(img_path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    # Processes images by scaling image RGB values
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)

    return tf.keras.applications.mobilenet_v2.preprocess_input(img_array_expanded_dims)


def New():
    layer = keras.layers.Dense(3)

    layer.build((None, 5))  # Create the weights

    print("weights:", len(layer.weights))
    print("trainable_weights:", len(layer.trainable_weights))
    print("non_trainable_weights:", len(layer.non_trainable_weights))


if __name__ == "__main__":
    print("Flower Classifier by Jarryd Stringfellow and Chase Dart")

    # Retrieve MobileNetV2 Data
    mobile = tf.keras.applications.mobilenet_v2.MobileNetV2()
    # mobile = tf.keras.applications.mobilenet_v2.MobileNetV2(
    #     input_shape=None,
    #     alpha=1.0,
    #     include_top=False,
    #     weights='imagenet',
    #     input_tensor=None,
    #     pooling=None,
    #     classes=1000,
    #     classifier_activation='softmax'
    # )

    # Single image test data
    # file = 'daisy/162362896_99c7d851c8_n.jpg'
    # file = 'dandelion/8223949_2928d3f6f6_n.jpg'
    # file = 'roses/99383371_37a5ac12a3_n.jpg'
    # file = 'sunflowers/39271782_b4335d09ae_n.jpg'
    # file = 'tulips/11746276_de3dec8201.jpg'

    # New()

    images = ()  # All images
    group = []  # Type of flower

    # Record the different groups
    for path, subdirs, files in os.walk(img_path):
        for name in subdirs:
            group += [name]
            img_path

    # print(group)

    # Record the images
    for file in os.listdir(img_path + "daisy/"):
        temp_tuple = (file, 'null')
        images += temp_tuple
        images = images[:-1]  # Remove null

    # print(images)

    # Cycle through all of the images
    for category in group:
        for picture in images:

            file = category + "/" + picture

            preprocessed_image = PrepareImage(file)
            predictions = mobile.predict(preprocessed_image)
            # Returns top 5 predictions
            results = imagenet_utils.decode_predictions(predictions)

            print(results)
