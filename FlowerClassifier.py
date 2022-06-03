# Imports PIL module
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing import image
from tensorflow import keras
import tensorflow as tf
import numpy as np
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
                              '../small_flower_dataset/')


def task_2():
    """
    Using the tf.keras.applications module download a pretrained MobileNetV2 network.
    """
    # Retrieve MobileNetV2 Data
    base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
        input_shape=None,
        alpha=1.0,
        # include_top=False,
        weights='imagenet',
        input_tensor=None,
        pooling=None,
        classes=1000,
        classifier_activation='softmax'
    )

    # Freeze the base_model
    base_model.trainable = False

    return base_model


def task_3(base_model):
    """
    Replace the last layer with a Dense layer of the appropriate shape given that there are 5
    classes in the small flower dataset.
    """
    inputs = keras.Input(shape=(224, 224, 5))
    # Change base_model to run in inference mode
    x = base_model(inputs, training=False)
    # Convert features of shape `base_model.output_shape[1:]` to vectors
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(5)(x)
    model = keras.Model(inputs, outputs)

    return model


def task_4(file):
    """
    Prepare your training, validation and test sets.
    Takes the individual image and prepares it by resizing and RGB scaling it for keras
    """
    # MobileNet expects 224x224 image sizes
    img = image.load_img(image_path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    # Processes images by scaling image RGB values
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)

    return tf.keras.applications.mobilenet_v2.preprocess_input(img_array_expanded_dims)


def task_5(model):
    """
    Compile and train your model with an SGD3
    optimizer using the following parameters
    learning_rate=0.01, momentum=0.0, nesterov=False.
    """

    pictures = ()  # All images
    categories = []  # Type of flower
    qty = 0  # Quantity of images in each folder

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
            qty += 1

        # Cycle through all of the pictures
        for picture in pictures:
            file = category + "/" + picture

            preprocessed_image = task_4(file)
            predictions = model.predict(preprocessed_image)
            # Returns top 5 predictions
            results = imagenet_utils.decode_predictions(predictions)

            print(results)

        pictures = ()

        # model.compile(optimizer=keras.optimizers.Adam(),
        #               loss=keras.losses.BinaryCrossentropy(
        #                   from_logits=True),
        #               metrics=[keras.metrics.BinaryAccuracy()])

        # model.fit(preprocessed_image, epochs=20,
        #           callbacks=..., validation_data=...)


def task_6():
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

    task_1()
    base_model = task_2()
    # model = task_3(base_model)
    # task_4() # Called from within task_5()
    task_5(base_model)  # Change to model to use task_3
    task_6()
    task_7()
    task_8()
