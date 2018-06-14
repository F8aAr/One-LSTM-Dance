# One-LSTM-Dance

## Generate a Drum pattern

Comment the training and uncomment the generation section in LSTMmodel.py

Choose a temperature (diversity) or just assign None

You can choose the number of bars you want to generate changing variable num_bars

Once you have generated a pattern from a random seed, you can fix the seed (if you like it). The index of the seed will be printed in console. So you will just have to assign that index to the variable seed_idx. Now you can test different temperatures for that same seed.

If you want to keep generating samples from random seeds, keep seed_idx = None
