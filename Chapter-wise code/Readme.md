# pytorch projects
The folder contains the projects of PyTorch Scholarship Challenge and Deep Learning Nanodegree from Udacity.

## Project 1: Flower Image Classification
The [flower_image_classification.ipynb]https://github.com/purvasingh96/Deep-learning-with-neural-networks/blob/master/Chapter-wise%20code/Code%20-%20PyTorch/2.%20Convolution%20Neural%20Networks/4.%20Transfer%20Learning/Transfer_Learning_predict_flowers.ipynb) contains the project Flower image classifier.
The 102 Category Flower [Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) from Visual Geometry Group, University of Oxford, is used.

The following **steps** are described:
* Preprocessing
* Transfer learning
* Saving and loading model checkpoint
* Inference and Validation

**Analysis:**
* Model used: VGG16
* Epochs trained: 2
* Validation accuracy: 74%
* Optimizer used: SGD (stochastic gradient descent)
* Loss used: CrossEntropyLoss
* Device used: cuda
* Comments: `fc` layer replacement with combination of linear layers with Dropout regularization

## Project 2: Neural Style Transfer 
The [style_transfer.ipynb](https://github.com/purvasingh96/Deep-learning-with-neural-networks/blob/master/Chapter-wise%20code/Code%20-%20PyTorch/2.%20Convolution%20Neural%20Networks/5.%20Style%20Transfer/Style_Transfer_via_pytorch.ipynb) decribes the style transfer implementation of [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) by *Leon A. Gatys et al.*

The **Abstract** of the paper is:
> Rendering the semantic content of an image in different styles is a difficult image processing task. Arguably, a major limiting factor for previous approaches has been the lack of image representations that explicitly represent semantic information and, thus, allow to separate image content from style. Here we use image representations derived from Convolutional Neural Networks optimised for object recognition, which make high level image information explicit. We introduce
A **Neural Algorithm of Artistic Style** that can separate and recombine the image content and style of natural images. The algorithm allows us to produce new images of high perceptual quality that combine the content of an arbitrary photograph with the appearance of numerous well-known artworks. Our results provide new insights into the deep image representations learned by Convolutional Neural Networks and demonstrate their potential for high level image synthesis and manipulation.

You can read the **summary** of the above paper [here](https://github.com/aleju/papers/blob/master/neural-nets/A_Neural_Algorithm_for_Artistic_Style.md).

