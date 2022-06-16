# Imports PIL module
from email.mime import base
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.applications import imagenet_utils
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
    fig = plt.figure(figsize=[25, 8])
    ax = fig.add_subplot(1, 3, 1)
    conf = ConfusionMatrixDisplay.from_estimator(
        model, X_train, Y_train, normalize='true', ax=ax)
    train_pred = model.predict(X_train)
    conf.ax_.set_title('Training Set Performance: ' +
                       str(sum(train_pred == Y_train)/len(Y_train)))

    ax = fig.add_subplot(1, 3, 2)
    conf = ConfusionMatrixDisplay.from_estimator(
        model, X_test, Y_test, normalize='true', ax=ax)
    test_pred = model.predict(X_test)
    conf.ax_.set_title('Test Set Performance: ' +
                       str(sum(test_pred == Y_test)/len(Y_test)))

    ax = fig.add_subplot(1, 3, 3)
    conf = ConfusionMatrixDisplay.from_estimator(
        model, X_val, Y_val, normalize='true', ax=ax)
    val_pred = model.predict(X_val)
    conf.ax_.set_title('Validation Set Performance: ' +
                       str(sum(val_pred == Y_val)/len(Y_val)))

    mins = np.min(X_train, 0)
    maxs = np.max(X_train, 0)
    xx, yy = np.meshgrid(np.arange(mins[0], maxs[0], 0.025),
                         np.arange(mins[1], maxs[1], 0.025))

    getScores(Y_val, val_pred, "Validation Scores")
    getScores(Y_test, test_pred, "Test Scores")


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

            # --------------------------------------------------------------
            y_data[qty, qty_class] = 1
            x_data[qty] = img_array_expanded_dims
            qty += 1
        qty_class += 1
        pictures = ()

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, random_state=1)
    x_test, x_val, y_test, y_val = train_test_split(
        x_test, y_test, test_size=0.5, random_state=1)
    print(x_train)
    print(y_train)
    return x_train, x_test, x_val, y_train, y_test, y_val


def task_5(model, x_train, x_test, x_val, y_train, y_test, y_val):
    """
    Compile and train your model with an SGD3
    optimizer using the following parameters
    learning_rate=0.01, momentum=0.0, nesterov=False.
    """
    # model.compile(optimizer=keras.optimizers.SGD( learning_rate = 0.01, momentum=0.0, nesterov = False), loss=None)
<<<<<<< HEAD
    model.compile(optimizer=keras.optimizers.SGD( learning_rate = 0.01, momentum=0.0, nesterov = False),
              loss= 'categorical_crossentropy',
              metrics=['accuracy'])     #sparse_categorical_crossentropy

    
    history = model.fit(x_train, y_train, epochs = 20, validation_data=(x_test, y_test))
=======
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.00001, momentum=0.0, nesterov=False),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])  # sparse_categorical_crossentropy

    # validation_data=(x_test, y_test),
    history = model.fit(x_train, y_train, epochs=40)
>>>>>>> e953790658eb587f05564dec7bdeda8112bbcec7
    return history


def task_6(history, model, x_train, x_test, x_val, y_train, y_test, y_val):
    """
    Plot the training and validation errors vs time as well as the training and validation
    accuracies.
    """
    # eval_model(model, x_train, y_train, x_test, y_test, x_val, y_val)

    # print(history.history.keys())

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

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
    y_train_arg = np.argmax(y_train, axis=1)
    Y_train_pred = np.argmax(model.predict(x_train), axis=1)
    cm = confusion_matrix(y_train_arg, Y_train_pred)
    disp = ConfusionMatrixDisplay.from_predictions(
        y_train_arg, Y_train_pred, normalize='true', ax=ax)
    ax.xaxis.set_ticklabels(
        ['daisy', 'dandelion', 'roses', 'sunflower', 'tulips'])
    ax.yaxis.set_ticklabels(
        ['daisy', 'dandelion', 'roses', 'sunflower', 'tulips'])
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Training Confusion Matrix')

    # Plotting for Test Data
    ax = fig.add_subplot(1, 3, 2)
    y_test_arg = np.argmax(y_test, axis=1)
    Y_train_pred = np.argmax(model.predict(x_test), axis=1)
    cm = confusion_matrix(y_test_arg, Y_train_pred)
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test_arg, Y_train_pred, normalize='true', ax=ax)
    ax.xaxis.set_ticklabels(
        ['daisy', 'dandelion', 'roses', 'sunflower', 'tulips'])
    ax.yaxis.set_ticklabels(
        ['daisy', 'dandelion', 'roses', 'sunflower', 'tulips'])
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Testing Confusion Matrix')

    # Plotting for Validation Data
    ax = fig.add_subplot(1, 3, 3)
    y_val_arg = np.argmax(y_val, axis=1)
    Y_val_pred = np.argmax(model.predict(x_val), axis=1)
    cm = confusion_matrix(y_val_arg, Y_val_pred)
    disp = ConfusionMatrixDisplay.from_predictions(
        y_val_arg, Y_train_pred, normalize='true', ax=ax)
    ax.xaxis.set_ticklabels(
        ['daisy', 'dandelion', 'roses', 'sunflower', 'tulips'])
    ax.yaxis.set_ticklabels(
        ['daisy', 'dandelion', 'roses', 'sunflower', 'tulips'])
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Validation Confusion Matrix')

    # ax.show()
    # mins = np.min(X_train, 0)
    # maxs = np.max(X_train, 0)
    # xx, yy = np.meshgrid(np.arange(mins[0], maxs[0], 0.025),
    #                  np.arange(mins[1], maxs[1], 0.025))

    # getScores(Y_val,val_pred, "Validation Scores")
    # getScores(Y_test, test_pred,"Test Scores")

    # plt.plot()


def task_7(model, x_train, x_test, x_val, y_train, y_test, y_val):
    """
    Experiment with 3 different orders of magnitude for the learning rate. Plot the results, draw
    conclusions.
    """

    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.001, momentum=0.0, nesterov=False),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])  # sparse_categorical_crossentropy

    # validation_data=(x_test, y_test),
    history_lr1 = model.fit(x_train, y_train, epochs=5)

    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.0001, momentum=0.0, nesterov=False),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])  # sparse_categorical_crossentropy

    # validation_data=(x_test, y_test),
    history_lr2 = model.fit(x_train, y_train, epochs=5)

    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.00001, momentum=0.0, nesterov=False),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])  # sparse_categorical_crossentropy

    # validation_data=(x_test, y_test),
    history_lr3 = model.fit(x_train, y_train, epochs=5)

    return history_lr1, history_lr2, history_lr3


def task_8(history1, history2, history3, model, x_train, x_test, x_val, y_train, y_test, y_val):
    """
    With the best learning rate that you found in the previous task, add a non zero momentum to
    the training with the SGD optimizer (consider 3 values for the momentum). Report how
    your results change.
    """
    plt.subplot(1, 2, 1)
    plt.plot(history1.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.plot(history2.history['accuracy'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.plot(history3.history['accuracy'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['lr 0.001', 'lr 0.0001', 'lr 0.00001'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history1.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.plot(history2.history['loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.plot(history3.history['loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['lr 0.001', 'lr 0.0001', 'lr 0.00001'], loc='upper left')


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
    x_train, x_test, x_val, y_train, y_test, y_val = task_4()
    print("\n---End of Task 4:---\n")

    print("\n---Start of Task 5:---\n")
    history = task_5(model, x_train, x_test, x_val, y_train, y_test, y_val)
    print("\n---End of Task 5:---\n")

    print("\n---Start of Task 6:---\n")
    task_6(history, model, x_train, x_test, x_val, y_train, y_test, y_val)
    print("\n---End of Task 6:---\n")

<<<<<<< HEAD
    # print("\n---Start of Task 7:---\n")
    # history1, history2, history3 = task_7(model, x_train, x_test, x_val, y_train, y_test, y_val)
    # print("\n---End of Task 7:---\n")

    # print("\n---Start of Task 8:---\n")
    # task_8(history1, history2, history3, model, x_train, x_test, x_val, y_train, y_test, y_val)
    # print("\n---End of Task 8:---\n")
=======
    print("\n---Start of Task 7:---\n")
    history1, history2, history3 = task_7(
        model, x_train, x_test, x_val, y_train, y_test, y_val)
    print("\n---End of Task 7:---\n")

    print("\n---Start of Task 8:---\n")
    task_8(history1, history2, history3, model, x_train,
           x_test, x_val, y_train, y_test, y_val)
    print("\n---End of Task 8:---\n")
>>>>>>> e953790658eb587f05564dec7bdeda8112bbcec7

    plt.show()
