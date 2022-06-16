from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.keras.preprocessing import image
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from tensorflow import keras
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

image_path = "/"  # Init for global path var with empty path


def getScores(y, pred, name):
    """
    Returns the various scores
    """
    print("--------------------- ", name, " --------------------")
    print("Accuracy score")
    print(accuracy_score(y, pred))
    print("F1 score")
    print(f1_score(y, pred, average='macro'))
    print("Recall")
    print(recall_score(y, pred, average='macro'))
    print("Precision")
    print(precision_score(y, pred, average='macro'))


def eval_model(model, X_train, Y_train, X_test, Y_test, X_val, Y_val):
    """
    Evaluates the model
    """
    # Create figure and subplots
    fig = plt.figure(figsize=[25, 8])
    ax = fig.add_subplot(1, 3, 1)

    # Training Set
    conf = ConfusionMatrixDisplay.from_estimator(
        model, X_train, Y_train, normalize='true', ax=ax)
    train_pred = model.predict(X_train)
    conf.ax_.set_title('Training Set Performance: ' +
                       str(sum(train_pred == Y_train)/len(Y_train)))

    # Test Set
    ax = fig.add_subplot(1, 3, 2)
    conf = ConfusionMatrixDisplay.from_estimator(
        model, X_test, Y_test, normalize='true', ax=ax)
    test_pred = model.predict(X_test)
    conf.ax_.set_title('Test Set Performance: ' +
                       str(sum(test_pred == Y_test)/len(Y_test)))

    # Validation Set
    ax = fig.add_subplot(1, 3, 3)
    conf = ConfusionMatrixDisplay.from_estimator(
        model, X_val, Y_val, normalize='true', ax=ax)
    val_pred = model.predict(X_val)
    conf.ax_.set_title('Validation Set Performance: ' +
                       str(sum(val_pred == Y_val)/len(Y_val)))

    getScores(Y_val, val_pred, "Validation Scores")
    getScores(Y_test, test_pred, "Test Scores")


def task_1():
    """
    Download the small flower dataset from Blackboard.
    Global declaration of filepath to dataset
    """
    global image_path
    # Direct path to folder location
    image_path = os.path.join(os.path.dirname(__file__),
                              'small_flower_dataset/')


def task_2():
    """
    Using the tf.keras.applications module download a pretrained MobileNetV2 network.
    """
    # Retrieve MobileNetV2 Data
    base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
        input_shape=(224, 224, 3),
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
    # Create a Dense layer of shape 5 for each flower type
    new_predictions = keras.layers.Dense(5)(base_model.layers[-2].output)
    new_model = keras.Model(inputs=base_model.inputs, outputs=new_predictions)

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

    x_data = np.empty((1000, 224, 224, 3))
    y_data = np.zeros((1000, 5))

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
            img = image.load_img(image_path + file, target_size=(224, 224, 3))
            img_array = image.img_to_array(img)

            # Processes images by scaling image RGB values
            img_array_expanded_dims = np.expand_dims(img_array, axis=0)

            y_data[qty, qty_class] = 1
            x_data[qty] = img_array_expanded_dims
            qty += 1

        qty_class += 1
        pictures = ()

    # Split the data into training and testing
    # The Training is 80%, Testing is 10% and Validation is 10%
    # X refers to the data and Y refers to the labels for the data
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, random_state=1)

    x_test, x_val, y_test, y_val = train_test_split(
        x_test, y_test, test_size=0.5, random_state=1)

    print(x_train)
    print(y_train)

    return x_train, x_test, x_val, y_train, y_test, y_val


def task_5(model, x_train, x_test, y_train, y_test):
    """
    Compile and train your model with an SGD3
    optimizer using the following parameters
    learning_rate=0.01, momentum=0.0, nesterov=False.
    """
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])  # sparse_categorical_crossentropy

    history = model.fit(x_train, y_train, epochs=20,
                        validation_data=(x_test, y_test))

    return history


def task_6(history, model, x_train, x_test, x_val, y_train, y_test, y_val):
    """
    Plot the training and validation errors vs time as well as the training and validation
    accuracies.
    """
    # Model accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # Model loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')

    fig = plt.figure(figsize=[25, 8])

    # Plotting for Training
    ax = fig.add_subplot(1, 3, 1)
    ax.xaxis.set_ticklabels(
        ['daisy', 'dandelion', 'roses', 'sunflower', 'tulips'])
    ax.yaxis.set_ticklabels(
        ['daisy', 'dandelion', 'roses', 'sunflower', 'tulips'])
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Training Confusion Matrix')

    # Plotting for Test Data
    ax = fig.add_subplot(1, 3, 2)
    ax.xaxis.set_ticklabels(
        ['daisy', 'dandelion', 'roses', 'sunflower', 'tulips'])
    ax.yaxis.set_ticklabels(
        ['daisy', 'dandelion', 'roses', 'sunflower', 'tulips'])
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Testing Confusion Matrix')

    # Plotting for Validation Data
    ax = fig.add_subplot(1, 3, 3)
    ax.xaxis.set_ticklabels(
        ['daisy', 'dandelion', 'roses', 'sunflower', 'tulips'])
    ax.yaxis.set_ticklabels(
        ['daisy', 'dandelion', 'roses', 'sunflower', 'tulips'])
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Validation Confusion Matrix')


def task_7(model, x_train, x_test, x_val, y_train, y_test, y_val):
    """
    Experiment with 3 different orders of magnitude for the learning rate. Plot the results, draw
    conclusions.
    """
    # Base model
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.001, momentum=0.0, nesterov=False),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])  # sparse_categorical_crossentropy

    history_lr1 = model.fit(x_train, y_train, epochs=2,
                            validation_data=(x_test, y_test))

    # changed learning_rate to 0.0001
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.0001, momentum=0.0, nesterov=False),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])  # sparse_categorical_crossentropy

    history_lr2 = model.fit(x_train, y_train, epochs=2,
                            validation_data=(x_test, y_test))

    # Changed momentum to 1
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=1, nesterov=False),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])  # sparse_categorical_crossentropy

    history_lr3 = model.fit(x_train, y_train, epochs=2,
                            validation_data=(x_test, y_test))

    return history_lr1, history_lr2, history_lr3


def task_8(history1, history2, history3):
    """
    With the best learning rate that you found in the previous task, add a non zero momentum to
    the training with the SGD optimizer (consider 3 values for the momentum). Report how
    your results change.
    """
    # Plot the results all on the one sheet
    plt.subplot(1, 2, 1)
    # Plot history 1 accuracy
    plt.plot(history1.history['accuracy'])
    plt.plot(history1.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')

    # Plot history 2 accuracy
    plt.plot(history2.history['accuracy'])
    plt.plot(history2.history['val_accuracy'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')

    # Plot history 3 accuracy
    plt.plot(history3.history['accuracy'])
    plt.plot(history3.history['val_accuracy'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')

    plt.legend(['Test: lr 0.001', 'Test: lr 0.001', 'Train: lr 0.0001',
               'Test: lr 0.0001', 'Train: Momentum 1', 'Test: Momentum 1'], loc='upper left')

    plt.subplot(1, 2, 2)
    # Plot history 1 loss
    plt.plot(history1.history['loss'])
    plt.plot(history1.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')

    # Plot history 2 loss
    plt.plot(history2.history['loss'])
    plt.plot(history2.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')

    # Plot history 3 loss
    plt.plot(history3.history['loss'])
    plt.plot(history3.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')

    plt.legend(['Test: lr 0.001', 'Test: lr 0.001', 'Train: lr 0.0001',
               'Test: lr 0.0001', 'Train: Momentum 1', 'Test: Momentum 1'], loc='upper left')


if __name__ == "__main__":
    """
    Main function
    Tasks can be commented out to filter by desired tests
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
    x_train, x_test, x_val, y_train, y_test, y_val = task_4()
    print("\n---End of Task 4:---\n")

    print("\n---Start of Task 5:---\n")
    history = task_5(model, x_train, x_test, y_train, y_test)
    print("\n---End of Task 5:---\n")

    print("\n---Start of Task 6:---\n")
    task_6(history, model, x_train, x_test, x_val, y_train, y_test, y_val)
    print("\n---End of Task 6:---\n")

    print("\n---Start of Task 7:---\n")
    history1, history2, history3 = task_7(
        model, x_train, x_test, x_val, y_train, y_test, y_val)
    print("\n---End of Task 7:---\n")

    print("\n---Start of Task 8:---\n")
    task_8(history1, history2, history3, model, x_train,
           x_test, x_val, y_train, y_test, y_val)
    print("\n---End of Task 8:---\n")

    # Plot results
    plt.show()
