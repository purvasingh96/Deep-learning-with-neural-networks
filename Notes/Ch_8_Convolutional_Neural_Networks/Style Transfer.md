# Style Transfer using CNNs

# Overview
One of the use cases of CNN is **style transfer.** Style transfer allows you to apply the style of one image to another image of your choice. For instance, in the below style-transfer example-<br>
<img src="./images/style_transfer/01.style_transfer.png" height=400 width=450></img>

We have applied **style of Hokusai wave** to the **content of cat.**<br>

## Separating Style and Content
Style transfer will look at 2 different images: **Style image** and **Content image**.<br>
Using a trained CNN, the model will try to find style from one image and content from another and would finally try to merge the two images to create a new third image.<br>
<img src="./images/style_transfer/02.content_and_style_image.png"></img>

## VGG19 and Content Loss
Below is an example of VGG19 CNN architecture which consists of multiple convolution and max-pooling layers, that uses style transfer method.<br><br>
<img src="./images/style_transfer/03. vgg19.png" height=280 width=650></img><br>

The first step to perform style transfer is to pass both style and content image through the CNN as below.<br><br>

1. First, when the network sees the content image, it will go through the feed-forward process, until it gets throgh convolution layer that is deep in the network. <br>
**Output** of this layer will be **content representation** of input image.<br>
<img src="./images/style_transfer/04. content_image_vgg19.png" height=280 width=650></img><br>

2. Next, when the CNN sees the style image, it extracts different features from multiple layers that represent the style of an image.<br>
<img src="./images/style_transfer/05. style_image_vgg19.png"></img><br>

3. Finally it will merge the content from step-1 and style from step-2 to form the **target image.**<br> 
<img src="./images/style_transfer/06. target_image.png"></img><br>






image source - https://www.udacity.com/course/deep-learning-nanodegree--nd101
