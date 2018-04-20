import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from keras import optimizers, Input, Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.initializers import TruncatedNormal
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, TimeDistributed, Concatenate, CuDNNLSTM, Dropout, \
    BatchNormalization, Reshape
from keras.utils import plot_model, to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

from keras_tf_multigpu.kuza55 import make_parallel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def plot_history(history):
    """
    Plot the accuracy and loss graph of the CNN and save in './plots'.

    :param history: object, containing the history of CNN.
    """
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    plt.savefig('./plots/acc_plot.png')
    plt.clf()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    plt.savefig('./plots/loss_plot.png')
    plt.clf()


def callbacks():
    """
    Callbacks used in CNN.

    :return: list, containing the callbacks.
    """
    checkpoint = ModelCheckpoint(filepath='./models/cnn_weights_{}.h5'.format(fold_index), monitor='val_acc',
                                 verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=40, verbose=1, min_lr=0.0001)
    early_stopping = EarlyStopping(monitor='val_acc', min_delta=0.01, patience=200)
    return [checkpoint, reduce_lr, early_stopping]


def cnn(channel=3):
    """
    Architecture and model of the CNN.

    :return: object, model of the CNN.
    """
    sgd = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=True)
    inputs = Input(shape=(16, 40, 80, 1))
    gaussian = TruncatedNormal(stddev=0.01, seed=None)

    pitch = TimeDistributed(
        Conv2D(filters=32, kernel_size=(32, 1), activation='relu', kernel_initializer=gaussian, name='conv_1'))(
        inputs)
    pitch = TimeDistributed(BatchNormalization())(pitch)
    pitch = TimeDistributed(MaxPooling2D(pool_size=(1, 80)))(pitch)
    pitch = TimeDistributed(Reshape((1, 9, -1)))(pitch)

    tempo = TimeDistributed(
        Conv2D(filters=32, kernel_size=(1, 60), activation='relu', kernel_initializer=gaussian, name='conv_2'))(
        inputs)
    tempo = TimeDistributed(BatchNormalization())(tempo)
    tempo = TimeDistributed(MaxPooling2D(pool_size=(40, 1)))(tempo)

    bass = TimeDistributed(
        Conv2D(filters=32, kernel_size=(13, 9), activation='relu', kernel_initializer=gaussian, name='conv_3'))(
        inputs)
    bass = TimeDistributed(BatchNormalization())(bass)
    bass = TimeDistributed(MaxPooling2D(pool_size=(4, 4)))(bass)
    bass = TimeDistributed(Reshape((1, 126, -1)))(bass)

    if channel == 2:
        concatenate = Concatenate(axis=3)([pitch, tempo])

        flatten = TimeDistributed(Flatten())(concatenate)
        dense = TimeDistributed(Dense(200, kernel_initializer=gaussian, activation='relu', name='dense_1'))(flatten)
        dropout = TimeDistributed(Dropout(0.5))(dense)

        lstm = CuDNNLSTM(100, kernel_initializer=gaussian, recurrent_initializer=gaussian, bias_initializer='zeros',
                         return_sequences=False, return_state=False, stateful=False, name='lstm_1')(dropout)
    else:  # channel == 3
        concatenate = Concatenate(axis=3)([pitch, tempo, bass])

        flatten = TimeDistributed(Flatten())(concatenate)
        dense = TimeDistributed(Dense(400, kernel_initializer=gaussian, activation='relu', name='dense_1'))(flatten)
        dropout = TimeDistributed(Dropout(0.5))(dense)

        lstm = CuDNNLSTM(200, kernel_initializer=gaussian, recurrent_initializer=gaussian, bias_initializer='zeros',
                         return_sequences=False, return_state=False, stateful=False, name='lstm_1')(dropout)

    predictions = Dense(10, activation='softmax', name='dense_3')(lstm)

    model = Model(inputs=inputs, outputs=predictions)
    model = make_parallel(model, gpu_count=2)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    plot_model(model, to_file='./plots/model_plot.png', show_shapes=True, show_layer_names=True)

    return model


def convert_to_cm_labels(test_y, prediction):
    """
    Convert the ground truths and the predicted labels

    :param test_y: 1D array, containing the labels of the test data.
    :param prediction: 2D array, containing the probability of the prediction.
    :return: 1D arrays, predicted labels and ground truth labels.
    """
    predicted_labels = np.zeros((prediction.shape[0]))
    ground_truth_labels = np.zeros((test_y.shape[0]))
    for instance in range(test_y.shape[0]):
        predicted_labels[instance] = np.argmax(prediction[instance, :])
        ground_truth_labels[instance] = np.argmax(test_y[instance, :])
    return predicted_labels, ground_truth_labels


def print_results(accuracy_list, history_list):
    """
    Print the results of the training.

    :param accuracy_list: list, containing the accuracy of each fold.
    :param history_list: list, containing the history object of each fold.
    """
    accuracy_mean = np.mean(accuracy_list)
    print("Mean accuracy: {:.03f}".format(accuracy_mean))
    accuracy_std = np.std(accuracy_list)
    print("Std accuracy: {:.03f}".format(accuracy_std))
    model_index = int(np.argmax(accuracy_list))
    print("Best model: {0}({1:.03f})".format(model_index, np.max(accuracy_list)))
    plot_history(history_list[model_index])


def check_options(argv):
    """
    Check for number of channels of the CNN.

    :param argv: [0] - script name (ignored), [1] - options = 2 | 3
    """
    if len(argv) < 2:
        handle_exit()
    elif len(argv) == 2:
        if argv[1] == '2':
            return 2
        elif argv[1] == '3':
            return 3
        else:
            handle_exit()


def handle_exit():
    """
    Exit handling upon incorrect system arguments.
    """
    print('Error: No such number of channels')
    print('Please enter the command in this format:')
    print("[SCRIPT] [OPTIONS 2 | 3]")
    print("\tpython train.py 3")
    sys.exit()


def load_data():
    """
    Load the data and labels of the train and test set.

    :return: train_X, train data; train_y, train labels; test_X, test data; test_y, test labels.
    """
    train_X = np.load('./models/train_X.npy')
    train_y = np.load('./models/train_y.npy')
    train_y = to_categorical(train_y, num_classes=10)
    test_X = np.load('./models/test_X.npy')
    test_y = np.load('./models/test_y.npy')
    test_y = to_categorical(test_y, num_classes=10)
    return train_X, train_y, test_X, test_y


if __name__ == '__main__':
    channel = check_options(sys.argv)
    epochs = 2000
    batch_size = 20

    train_X, train_y, test_X, test_y = load_data()

    # K-Fold
    n_splits = 10
    kfold = KFold(n_splits=n_splits, shuffle=True)
    fold_index = 0
    accuracy_list = []
    history_list = []
    for train, val in kfold.split(train_X, train_y):
        model = cnn(channel)

        history = model.fit(train_X[train], train_y[train], validation_data=(train_X[val], train_y[val]), epochs=epochs,
                            batch_size=batch_size, verbose=2, callbacks=callbacks())
        history_list.append(history)

        model.load_weights('./models/cnn_weights_{}.h5'.format(fold_index))

        accuracy = (model.evaluate(test_X, test_y, verbose=2))[1]
        accuracy_list.append(accuracy)

        prediction = model.predict(test_X, verbose=2)

        predicted_labels, ground_truth_labels = convert_to_cm_labels(test_y, prediction)

        cm = confusion_matrix(ground_truth_labels, predicted_labels)
        print("In the {0} fold, the classification accuracy is {1:.03f}".format(fold_index, accuracy_list[fold_index]))
        print("and the confusion matrix is: ")
        print(cm, end='\n\n')

        fold_index += 1

    print_results(accuracy_list, history_list)
