# Autoencoders

## Overview
Autoencoders are neural network that is trained to attemp to copy its input
to its output. <br>
The network may be viewed as consisting of two parts: an
encoder function h = f (x) and a decoder that produces a reconstruction r = g(h).
<br><br>
Key component of an autoencoder is its ability to compress such that image data is still retained.

## Architecture

Below is an architecture of autoencoder and its 2 components  -  **encoder and decoder**<br>
<img src="./images/autoencoder_architecture.png"></img>
PyTorch implementation of a simple auto-encoder can be found here -  

## Undercomplete Autoencoders
We don't want autoencoders to completely copy the contents of input image to the output image. This might seem useless. We just want that autoencoder **learns important features** of features instead of entire input.<br><br>

One way to obtain useful features from the autoencoder is to constrain h to
have smaller dimension than x. An autoencoder whose code dimension is less
than the input dimension is called **undercomplete.**<br><br>

Autoencoders with **nonlinear encoder functions f and nonlinear decoder functions g** can learn a more powerful nonlinear generalization of PCA.
