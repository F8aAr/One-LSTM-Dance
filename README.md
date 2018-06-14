# One-LSTM-Dance

## Generate a Drum pattern

Comment the training and uncomment the generation section in LSTMmodel.py

Choose a temperature (diversity) or just assign None

You can choose the number of bars you want to generate changing variable num_bars

Go to generate_DrumLoop function and load the weights...  model.load_weights("weights_test.hdf5")
* make sure weights_test.hdf5 file is in the project directory

Once you have generated a pattern from a random seed, you can fix the seed (if you like it). The index of the seed will be printed in console. So you will just have to assign that index to the variable seed_idx. Now you can test different temperatures for that same seed.

If you want to keep generating samples from random seeds, keep seed_idx = None

## Train a new model

For training a new model comment the Generation section and uncomment the training. 

You can choose the sequence length(in bars) of the batches you want to input to the network in the Prepare Data section. In order to set semiredundant sequences set a step(now is set to 32). 

Go to build_model function and adjust your desire architecture.

Run training(LSTMmodel.py)

* We have uploaded the Text_corpus, make sure it is in the project directory and assign it to the variable path in prepare data section....  path = 'GS_QUAD_MDF_sel.txt' 
