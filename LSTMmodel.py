import numpy as np
import itertools
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Dropout
from keras.utils.vis_utils import plot_model
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
from matplotlib import pyplot
import pydot
from midi_drums_utils import removeBar, repeatBar, addBar, conv_text_to_midi,durBars, separate_words
import os
from graphviz import *
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'



"""

        - We read the text file and create 1D vectors in order to input to the model -

"""

def prepare_data(path=None,seq_len = None,step = None):

    if path == None:
        print('No path  or  wrong path provided')
        return
    else:
        text = open(path).read()
        print('corpus length:', len(text))
        open(path).close()

        #Maxlen change the length of the sequence you input the network for predicting 1 note


        maxlen = 16*seq_len   # Con 'BAR' es step*2 + 1 (17) ----------- Sin 'BAR' es step*2 (16)

        if step==None:
            step = maxlen

        beat_seq = text.split(' ')
        beat_seq = [ele for ele in beat_seq if ele not in ['BAR','']]
        words = set(beat_seq)
        text = beat_seq

        #Word to integer and Integer to word dictionaries
        word_indices = dict((note, num) for num, note in enumerate(words))
        indices_word = dict((num, note) for num, note in enumerate(words))

        num_words = len(word_indices)
        print('Nº words:', num_words)
        print(len(text))


        # I/O placeholders
        sequences = []
        next_words = []
        X = []
        y = []

        print('nb sequences:', len(sequences))
        print('Vectorization...')
        print('Reshaping...Normalizing...\n')


        #Create sequences (can be semiredundant)
        for i in range(0, len(text) - maxlen, step):
            sequences.append(text[i: i + maxlen])
            next_words.append(text[i + maxlen])

        for j, seq in enumerate(sequences):
            X.append([word_indices[word] for word in seq])
            y.append(word_indices[next_words[j]])

        num_seq = len(X)


        #reshaping input embedding and turning output to categorical
        X = np.reshape(X, (num_seq, maxlen, 1))
        y = np_utils.to_categorical(y)

        #Normalize input
        X = X/ float(num_words)

        print('Total nº input samples:', X.shape[0])
        print('Length of sequence(nº Timesteps) :', X.shape[1])
        print('Dimension:' ,X.shape[2], '\n')

        return (X,y,num_words,words)

'''

        - Build Architecture of model -
 
'''
def build_model(input, output, num_words, plot_network=False,generate = False):


    if not generate:
        model = Sequential()
        model.add(LSTM(256, input_shape = (input.shape[1], input.shape[2]), return_sequences = True))
        model.add(Dropout(0.3))
        model.add(LSTM((512), return_sequences = True))
        model.add(Dropout(0.3))
        model.add(LSTM((512), return_sequences=False))
        model.add(Dropout(0.3))
        model.add(Dense(num_words))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'] )

        #Schematic of the model
        if plot_network:
            plot_model(model, to_file='modelGen.png', show_shapes=True)

        return model
    else:
        model = Sequential()
        model.add(LSTM(256, input_shape=(input.shape[1], input.shape[2]), return_sequences=True))
        model.add(LSTM((512), return_sequences=True))
        model.add(LSTM((512), return_sequences=False))
        model.add(Dense(num_words))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        # Schematic of the model
        if plot_network:
            plot_model(model, to_file='modelGen.png', show_shapes=True)

        return model

"""

        - Train your model -

"""
def training(input, output, num_words, words):

    model = build_model(input, output, num_words,True,False)



    filepath = "output/weights-_seq128_LSTM512-256_batch16_Step8_-{epoch:02d}-{loss:.4f}.hdf5" #weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5
    checkpoint = ModelCheckpoint(
        filepath , monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    early_stop = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=3,
        verbose=0,
        mode='auto'
    )                                                # Big batches and small learning rate --> small patience

    callbacks_list = [checkpoint]



    history = model.fit(input, output, epochs=30, batch_size=32, callbacks=callbacks_list)

    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('model train')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.show()

    history = model.fit(input, output, epochs=30, batch_size=16, callbacks=callbacks_list)

    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('model train')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.show()
    """
    
    history = model.fit(input, output, epochs=1000, batch_size=16, callbacks=callbacks_list)

    # We can plot the loss of our model

    pyplot.plot(history.history['loss'])

    pyplot.title('model train')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train'], loc='upper right')
    pyplot.show(block=False)
    
    
    
    history = model.fit(input, output, epochs=70, batch_size=16, callbacks=callbacks_list)

    pyplot.plot(history.history['loss'])

    pyplot.title('model train')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train'], loc='upper right')
    pyplot.show(block=False)


    history = model.fit(input, output, epochs=70, batch_size=16, callbacks=callbacks_list)

    pyplot.plot(history.history['loss'])

    pyplot.title('model train')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train'], loc='upper right')
    pyplot.show(block=False)

    history = model.fit(input, output, epochs=300, batch_size=16, validation_split=0.33, callbacks=callbacks_list)

    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('model train')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.show()
    history = model.fit(input, output, epochs=300, batch_size=16, validation_split=0.33, callbacks=callbacks_list)

    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('model train')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.show()
"""



"""

        - Sampling multinomial distribution -

"""


def sample(a, temperature):
	# helper function to sample an index from a probability array
    a = np.asarray(a).astype('float64')
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))

    return np.argmax(np.random.multinomial(1, a, 1))


"""

        - Generate drums -

"""
def generate_DrumLoop(input, output, words,num_bars,diversity = None,filename=None,seed_idx = None):

    num_bars = num_bars*16 #Con 'BAR' 17------- Sin 'BAR' 16   ||||   Si longitud secuencia es 64 generar 4 BARs si 16 --> 8
    if seed_idx == None:
        sample_idx = np.random.randint(0,len(input)-num_bars-1)#-num_bars
    else:
        sample_idx = seed_idx

    indices_word = dict((num, note) for num, note in enumerate(words))
    num_words = len(indices_word)

    seed_word = input[sample_idx] # 17,1

    ## We save seed
    seed_word_out = seed_word*num_words
    seed_word_out = [indices_word[int(word)] for word in seed_word_out]
    file = open('Seed'+filename, "w")
    for item in seed_word_out:
        file.write("%s " % str(item))
    file.close()

    """
    ## We save idx
    file = open('Seed_idx_' + filename, "w")
    file.write("%s " % str(sample_idx))
    file.close()
    """


    seed_word_int = seed_word*num_words
    seed_word_int = np.int_(seed_word_int)
    seed_word_list = []



    # TESTING
    for num in seed_word_int:
        seed_word_list.append(num)

    # This converts to normal list (Flattens list)
    seed_word_list = list(itertools.chain.from_iterable(seed_word_list))



    seed_word_temp = []
    for i, note in enumerate(seed_word_list):
        for key in indices_word:
            if seed_word_list[i] == key:
                seed_word_temp.append(indices_word[key])



    pred_out = []
    model = build_model(input, output, num_words,True,True)

    # We load the weights of the model for generating
    model.load_weights("weights_test.hdf5")

    in_seed_temp = np.zeros((1, (len(seed_word) + 1), 1))
    in_seed = np.reshape(seed_word, (len(seed_word), 1), 1)  # in_seed = np.reshape(in_seed,(1,1),1)
    in_seed = np.expand_dims(in_seed, axis=0)
    #in_seed = in_seed/float(num_words) #CAMBIO
    for note_index in range(num_bars):


        guesses = model.predict(in_seed, verbose= 0)[0]

        # In case we use the temperature parameter

        if diversity == None:
            pred_idx = np.argmax(guesses)
        else:
            pred_idx = sample(guesses,diversity)

        pred_note = indices_word[pred_idx]


        in_seed_temp = np.zeros((1,(np.size(in_seed)+1),1))
        in_seed_temp[0,0:-1,0] = in_seed[0,:,0]

        #in_seed_temp = in_seed_temp + in_seed

        in_seed_temp = np.delete(in_seed_temp, 0, axis = 1)
        in_seed_temp[0,(np.size(in_seed)-1),0] = pred_idx/float(num_words)

        in_seed[0,:,0] = in_seed_temp[0,:,0]
        #print(np.shape(in_seed))
        #print(in_seed)

        pred_out.append(pred_note)


    print(pred_out)

    file = open(filename,"w")
    for item in pred_out:
        file.write("%s " % item)
    file.close()

    print('\n\nSample index:  '+str(sample_idx))


    return




"""
              - EXECUTE -
              




            - Prepare Data -   
"""

#separate_words('GS_QUAD_MDF_sel.txt')
path = 'GS_QUAD_MDF_sel.txt'
#path = repeatBar(path,repeatbar=16,returnFilename=True)

print('\nNumber of bars in the corpus: ', durBars(path))
new_path = removeBar(path,True)
step = 32
seq_len = 8 #number of bars 1 means 16 2 means 32...
(X,y,num_cat,words) = prepare_data(new_path,seq_len,step)


"""

            - Train the model -

"""


#training(X, y, num_cat,words)



"""

            - Generate Drum track -

"""


num_bars = 8  # Number of bars generated||seq_len is the len used to predict each sample
diversity = None# Hyperparameter 0.9, 1.0, 1.1.... or None 1.5 mola
seed_idx =None# None will take new seed
filename = 'Generated_pattern.txt'
generate_DrumLoop(X,y,words,num_bars,diversity,filename,seed_idx)


addBar('Seed'+filename)
conv_text_to_midi('Seed'+filename)
addBar(filename)
conv_text_to_midi(filename)
