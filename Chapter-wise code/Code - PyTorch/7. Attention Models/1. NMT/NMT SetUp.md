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
<img src="./images/11. NMT setup - german.png" width="50%"></img><br><br>
5. Keep track of index mappings with word2index and index2word mappings.
5. Use start-of-sentence `<SOS>` and end-of-sentence `<EOS>` tokens to represent the same.
