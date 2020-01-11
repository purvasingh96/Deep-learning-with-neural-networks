# Recurrent Neural Networks (RNNs)

## Overview

Consider an example given below where the neural network has to identify whether the white animal is a wolf/dog/goldfish. Now, based
on the probability distribution, the neural network predicts that the animal is a **Dog**, which is wrong, the correct prediction is
**wolf.** <br><br>
<img src="./images/01. traditional neural network.png" width="300px"></img>

Now in order to correct this, we take help of RNNs, where we take help of previous images shown to neural network, to hint to us that
the current image is that of a wolf.<br><br>
<img src="./images/02. RNN.png" width="450px" height="250px"></img>

## How do RNNs work?

In the example above, each output from previous neural network is fed as an input to current neural network which will in turn improve 
our results.<br>

## Drawbacks of RNN
* In the example above, suppose the bear appeared a while ago and immediate pictures predicted by a neural network are that of a tree and a squirrel. Based on the previous 2 images, we don't really know if the current image is that of a dog/wolf. Hence information that given picture is wolf comes all the way back from bear.

* As the information gets passed on to each layer of neural network it gets squished by **sigmoid functions** and training network from all the way back leads to problems such as **vanishing gradient**.

* RNNs have **short-term memory.**<br><br>
<img src="./images/04. Drawbacks_of_RNN.png" width="500px" height="250px"></img>

Next -  [LSTM](https://github.com/purvasingh96/Deep-learning-with-neural-networks/blob/master/Notes/Ch_9_Recurrent_Neural_Networks/LSTM.md)

