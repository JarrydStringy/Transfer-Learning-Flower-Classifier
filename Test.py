# Imports PIL module
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing import image
from tensorflow import keras
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# open method used to open different extension image file
filepath = os.path.join(os.path.dirname(__file__),
                        '../small_flower_dataset/daisy/5794839_200acd910c_n.jpg')
im = Image.open(filepath,)

# This method will show image in any image viewer
im.show()


"""
Flower Classifier

Created by: Jarryd Stringfellow
"""

# setup

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


def task_1():
    """
    Download the small flower dataset from Blackboard.

    """

    categories = []  # Type of flower
    images = ()  # All images

    # Record the different groups
    for subdirs in os.walk(img_path):
        for name in subdirs:
            categories += [name]

    # print(group)

    # Record the images
    for file in os.listdir(img_path + "daisy/"):
        temp_tuple = (file, 'null')
        images += temp_tuple
        images = images[:-1]  # Remove null

    # print(images)

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

    # Cycle through all of the images
    for category in categories:
        for item in images:

            file = category + "/" + item

            preprocessed_image = PrepareImage(file)
            predictions = mobile.predict(preprocessed_image)
            # Returns top 5 predictions
            results = imagenet_utils.decode_predictions(predictions)

            print(results)


def task_2():
    """
    Using the tf.keras.applications module download a pretrained MobileNetV2 network.
    """


def task_3():
    """
    Replace the last layer with a Dense layer of the appropriate shape given that there are 5
    classes in the small flower dataset.
    """


def task_4():
    """
    Prepare your training, validation and test sets.
    """


def task_5():
    """
    Compile and train your model with an SGD3
    optimizer using the following parameters
    learning_rate=0.01, momentum=0.0, nesterov=False.
    """


def task_6():
    """
    Plot the training and validation errors vs time as well as the training and validation
    accuracies.
    """


def task_7():
    """
    Experiment with 3 different orders of magnitude for the learning rate. Plot the results, draw
    conclusions.
    """


def task_8():
    """
    With the best learning rate that you found in the previous task, add a non zero momentum to
    the training with the SGD optimizer (consider 3 values for the momentum). Report how
    your results change.
    """


def New():
    """
    Testing
    """
    layer = keras.layers.Dense(3)

    layer.build((None, 5))  # Create the weights

    print("weights:", len(layer.weights))
    print("trainable_weights:", len(layer.trainable_weights))
    print("non_trainable_weights:", len(layer.non_trainable_weights))


if __name__ == "__main__":
    """
    Main function
    """
    print("========================================================")
    print("Flower Classifier by Jarryd Stringfellow and Chase Dart")
    print("========================================================")

    task_1()
    task_2()
    task_3()
    task_4()
    task_5()
    task_6()
    task_7()
    task_8()

    # New()
