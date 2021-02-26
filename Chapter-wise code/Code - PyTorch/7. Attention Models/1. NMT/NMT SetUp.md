# Setup for Machine Translation

## Data in NMT

Below we have the data sequence in English, *I'm hungry*, and on the right, the corresponding German equivalent. 
Further down we have, *I watch the soccer game*, and, the corresponding German equivalent. 
We are going to have a great many of these inputs. One thing to note here is that the data set used is not entirely clean.

<img src="./images/10. data in NMT.png" width="50%"></img><br><br>

## Pre-requisites

1. *Input*: Take English sentence as input.
2. *Tokenization*: State-of-the-art models use pre-trained word vectors, else, represent words with one-hot vectors to create the input.
3. *Padding*: Pad the tokenized sequence to make the inputs of equal length.<br><br>
<img src="./images/11. NMT setup-english.png" width="50%"></img><br><br>
4. Repeat steps 1-3 for the German sentences as well.<br><br>
<img src="./images/12. NMT setup - german.png" width="50%"></img><br><br>
5. Keep track of index mappings with word2index and index2word mappings.
5. Use start-of-sentence `<SOS>` and end-of-sentence `<EOS>` tokens to represent the same.

## Training NMT

### Teacher Forcing

Let us assume we want to train an image captioning model, and the ground truth caption for an  image is “Two people reading a book”. Our model makes a mistake in predicting the 2nd word and we have “Two” and “birds” for the 1st and 2nd prediction respectively.
1. *Without Teacher Forcing*, we would feed “birds” back to our RNN to predict the 3rd word. Let’s say the 3rd prediction is “flying”. Even though it makes sense for our model to predict “flying” given the input is “birds”, it is different from the ground truth.
<br><img src="./images/13. No teacher forcing.png"></img><br>
2. *With Teacher Forcing*, we would feed “people” to our RNN for the 3rd prediction, after computing and recording the loss for the 2nd prediction.
<br><img src="./images/14. with teacher forcing.png"></img><br>