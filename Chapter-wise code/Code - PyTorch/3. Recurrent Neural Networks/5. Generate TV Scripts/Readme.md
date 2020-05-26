# Generate-TV-Scripts
Udacity Deep Learning Nanodegree Project #3.

Objective: Generate your own Seinfeld TV scripts using RNNs. The input data was Seinfeld dataset of scripts from 9 seasons. The Neural Network you'll build will generate a new ,"fake" TV script, based on patterns it recognizes in this training data.

What I learnt:
1. Implemening Pre-processing Functions on the dataset: 

        * Lookup Table
        * Tokenize Punctuation
        
2. Building the Neural Network:

        * Batching the data
        * Creating dataloaders
        * Initializing RNN model and defining layers
        * Forward propagation of the neural network
        * Initialize the hidden state of an LSTM/GRU
        * Apply forward and back propagation
        
3. Traning the Neural Network:

        * Setting the hyperparameters for optimal loss values.
        
4. Generating new TV scripts using the trained model.         
