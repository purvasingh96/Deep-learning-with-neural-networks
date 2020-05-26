# Attention

## Sequence to Sequence Models (RNNs) 

### How RNNs work?

In order to better understand RNNs, let us take a look at an example of a machine translation model.<br> Here the model has 2 parts : **Encoder and Decoder**. We feed in the tokenzied input to encoder one-by-one. For the below example, we first feed the word **Comment** to encoder, which geneates it corresponding **hidden layer #1**. Next, we feed encoder, the 2nd word **allez**. The **hidden layer #2** is formed using **hidden layer #1 plus tokenzied word** and so on. Finally the **hidden layer #3** generated for the last word is what is fed as **context** to decoder.<br><br>
<img src="./images/sequence_to_sequence.png"></img>

### Drawback of RNN

The drawback of RNN or any sequence model is that it is confined to sending a single vector, no matter how long or short the input sequence is. Chosing the size for this vector makes the model have problems with long input sequences. In this case, one may suggest to use large sizes of hidden layers, but in this case, your model will overfit for short sequences. This is the problem that **Attention** solves.


## Applications of Attention

1. Machine translation
2. Document Summarization
3. Dailgoue Exchange
4. Image Caption Generator
