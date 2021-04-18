import sys
import argparse
import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.layers import Bidirectional
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

def train_network(note_path, n_epoch):
    notes = get_notes(note_path)
    
    n_vocab = len(set(notes))

    network_input, network_output = prepare_sequences(notes, n_vocab)

    model = create_network(network_input, n_vocab)

    train(model, network_input, network_output, n_epoch)

def get_notes(note_path):
    with open('notes/'+str(note_path), 'rb') as f:
        x = pickle.load(f)

        return x

def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 50

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)

def create_network(network_input, n_vocab):

    model = Sequential()
    model.add(Bidirectional(LSTM(
        128,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    )))
    model.add(LSTM(128, return_sequences=True, recurrent_dropout=0.3,))
    model.add(LSTM(128))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

def train(model, network_input, network_output, n_epoch):
    """ train the neural network """
    filepath = "./trained_weights/weight_improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=n_epoch, batch_size=128, callbacks=callbacks_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the Bi-LSTM model.')
    parser.add_argument('string', metavar='note_path', type=str, nargs='?',
                    help='the path name of the training set (notes)')
    parser.add_argument('int', metavar='number of epochs', type=int, nargs='?',
                    help='the number of epochs for training')
    args = parser.parse_args()
    
    try:
        note_path = sys.argv[1:][0]
        n_epoch = int(sys.argv[1:][1])
        train_network(note_path, n_epoch)
    except:
        print("Please specify the \"training set\" and the \"number of epochs\" as two arguments")


    


