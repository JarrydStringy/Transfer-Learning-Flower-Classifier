# Imports PIL module
from email.mime import base

from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing import image
from tensorflow import keras
import tensorflow as tf
import numpy as np
from imageio import imread
import os
import matplotlib.pyplot as plt

image_path = "/"  # Init for global path var with empty path
    
def task_1():
    """
    Download the small flower dataset from Blackboard.
    Global declaration of filepath to dataset
    """
    global image_path
    image_path = os.path.join(os.path.dirname(__file__),
                              'small_flower_dataset/')


def task_2():
    """
    Using the tf.keras.applications module download a pretrained MobileNetV2 network.
    """
    # Retrieve MobileNetV2 Data
    base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
        input_shape=(224,224,3),
        alpha=1.0,
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        pooling=None,
        classes=1000,
        classifier_activation='softmax'
    )

    # Freeze the base_model
    base_model.trainable = False
    base_model.summary()

    return base_model


def task_3(base_model):
    """
    Replace the last layer with a Dense layer of the appropriate shape given that there are 5
    classes in the small flower dataset.
    """

    new_predictions = keras.layers.Dense(5)(base_model.layers[-2].output)
    new_model = keras.Model(inputs = base_model.inputs, outputs = new_predictions)

    new_model.summary()

    return new_model


def task_4():
    """
    Prepare your training, validation and test sets.
    Takes the individual image and prepares it by resizing and RGB scaling it for keras
    """
    pictures = ()  # All images
    categories = []  # Type of flower
    qty = 0  # Quantity of images in each folder
    qty_class = 0

    x_data = np.empty((1000,224,224,3))
    y_data = np.empty(1000, dtype=np.float32)

    # Record the different groups
    for path, subdirs, files in os.walk(image_path):
        for name in subdirs:
            categories += [name]

    # Record the pictures
    for category in categories:
        for file in os.listdir(image_path + category + '/'):
            temp_tuple = (file, 'null')
            pictures += temp_tuple
            pictures = pictures[:-1]  # Remove null

        # Cycle through all of the pictures
        for picture in pictures:
            file = category + "/" + picture

            # MobileNet expects 224x224 image sizes
            img = image.load_img(image_path + file, target_size=(224, 224))
            img_array = image.img_to_array(img)

            # Processes images by scaling image RGB values
            img_array_expanded_dims = np.expand_dims(img_array, axis=0)

            #--------------------------------------------------------------
            y_data[qty] = qty_class
            x_data[qty] = img_array_expanded_dims
            qty += 1
        qty_class += 1
        pictures = ()

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.4, random_state=1)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=1)
    
    # print(x_data.shape())
    # print(y_data.shape())
    # MobileNet expects 224x224 image sizes
    # img = image.load_img(image_path + file, target_size=(224, 224))
    # img_array = image.img_to_array(img)

    # # Processes images by scaling image RGB values
    # img_array_expanded_dims = np.expand_dims(img_array, axis=0)

    return x_train, x_test, x_val, y_train, y_test, y_val


def task_5(model, x_train, x_test, x_val, y_train, y_test, y_val):
    """
    Compile and train your model with an SGD3
    optimizer using the following parameters
    learning_rate=0.01, momentum=0.0, nesterov=False.
    """

    # model.compile(optimizer=keras.optimizers.SGD( learning_rate = 0.01, momentum=0.0, nesterov = False), loss=None)
    model.compile(optimizer=keras.optimizers.SGD( learning_rate = 0.01, momentum=0.0, nesterov = False),
              loss='sparse_categorical_crossentropy')


    model.fit(x_val, y_val, epochs = 10)


def task_6(model, x_train, x_test, x_val, y_train, y_test, y_val):
    """
    Plot the training and validation errors vs time as well as the training and validation
    accuracies.
    """
    
    # plt.plot()


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


if __name__ == "__main__":
    """
    Main function
    """
    print("========================================================")
    print("Flower Classifier by Jarryd Stringfellow and Chase Dart")
    print("========================================================")

    print("\n---Start of Task 1:---\n")
    task_1()
    print("\n---End of Task 1:---\n")

    print("\n---Start of Task 2:---\n")
    base_model = task_2()
    print("\n---End of Task 2:---\n")

    print("\n---Start of Task 3:---\n")
    model = task_3(base_model)
    print("\n---End of Task 3:---\n")

    print("\n---Start of Task 4:---\n")
    x_train, x_test, x_val, y_train, y_test, y_val = task_4() # Called from within task_5()
    print("\n---End of Task 4:---\n")

    print("\n---Start of Task 5:---\n")
    task_5(model, x_train, x_test, x_val, y_train, y_test, y_val)  # Change to model to use task_3
    print("\n---End of Task 5:---\n")

    task_6(model, x_train, x_test, x_val, y_train, y_test, y_val)
    # task_7()
    # task_8()
