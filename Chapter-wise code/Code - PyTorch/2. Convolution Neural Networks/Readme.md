# Convolution Neural Networks

## Image Classification Pipeline 
<img src = "img/01.image_classificatio_steps.png"></img>

## MLPs v/s CNNs
| MLP                             | CNN                                                     |
|---------------------------------|---------------------------------------------------------|
| Only use fully-connected layers | Makes use of fully as well as sparsely connected layers |
| Only accepts vector as input    | Also accepts matrices as input                          |



### MLPs 
In a MLP, every hidden node needs to be connected to every pixel in input image. This accounts for a lot of redundancy.<br><br>
<img src = "img/02. MLP.png"></img>

### CNNs
Instead of every node keeping information of every pixel of input image, we divide the image into 4 regions - red, green, blue and yellow. Then each hidden node can be connected to only the pixels in one of these 4 regions as depicted below. <br><br>
<img src = "img/03. CNNs'.png"></img>

## Installation

To install opencv follow either of the below 2 methods -
1. [Install OpenCV on Linux](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html)
2. [Install OpenCV on Windows](https://www.learnopencv.com/install-opencv3-on-windows/)
3. If you have Anaconda Python installed, just run the below command - <br>
```
conda install opencv
```

Note - You can validate installation of opencv by running, `import opencv` 

