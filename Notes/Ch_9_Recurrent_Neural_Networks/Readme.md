# Recurrent Neural Networks (RNNs)

## Overview

Consider an example given below where the neural network has to identify whether the white animal is a wolf/dog/goldfish. Now, based
on the probability distribution, the neural network predicts that the animal is a **Dog**, which is wrong, the correct prediction is
**wolf.** <br>

Now in order to correct this, we take help of RNNs, where we take help of previous images shown to neural network, to hint to us that
the current image is that of a wolf.

## How do RNNs work?

In the example above, each output from previous neural network is fed as an input to current neural network which will in turn improve 
our results.<br>

## Drawbacks of RNN
In the example above, suppose the bear appeared a while ago and immediate pictures predicted by a neural network are that of a tree and 
a squirrel. Based on the previous 2 images, we don't really know if the current image is that of a dog/wolf. Since, the information 
