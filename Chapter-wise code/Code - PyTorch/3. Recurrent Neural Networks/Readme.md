# Hyper-parameter tuning in RNNs

Please refer to this paper for more detailed explaination : [Optimal Hyperparameters for Deep LSTM-Networks for Sequence
Labeling Tasks](https://github.com/purvasingh96/Deep-learning-with-neural-networks/blob/master/Chapter-wise%20code/Code%20-%20PyTorch/3.%20Recurrent%20Neural%20Networks/data/hyper_parameter_tuning.pdf)


## Overview
1. pretrained word embeddings usually perform the best.
2. Adam optimizer with Nestrov momentum  yields the highest performance and converges the fastest.
3. Gradient clipping does not help to improve the performance.
4. A large improvement is observed when using gradient normalization.
5. Two stacked recurrent layers usually performs best.
6. The impact of the number of recurrent units is rather small.
7. Around 100 recurrent units per LSTM-network appear to be a good rule of thumb.
8. Optimizer :  SGD has troubles to navigate ravines and at saddle points and is sensitive to learning rate. To eliminate
the short comings of SGD, other gradient-based optimization algorithms have been proposed - Adagrad, Adadelta, RMSProp, Adam and Nadam (an Adam variant that incorporates Nesterov momentum)
