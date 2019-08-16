Table of Contents
===================
- [Convolutional Neural Networks](#convolutional-neural-networks)
- [Convolution Operation](#convolution-operation)
  * [Example Demonstrating Convolution Operation](#example-demonstrating-convolution-operation)
  * [Properties of Convolution Operation and Cross-Correlation](#properties-of-convolution-operation-and-cross-correlation)
    + [Commutative Property](#commutative-property)
    + [Cross-Correlation](#cross-correlation)
  * [Toeplitz Matrix](#toeplitz-matrix)
  * [Block-Circulant and Doubly-Block-Circulant Matrix](#block-circulant-and-doubly-block-circulant-matrix)
- [Motivation](#motivation)
  * [Sparse Interactions](#sparse-interactions)
  * [Parameter Sharing](#parameter-sharing)
    + [Equivariance](#equivariance)
- [Pooling](#pooling)
  * [Inputs having Variable Size](#inputs-having-variable-size)
  * [Learned Invariances](#learned-invariances)
- [Convolution and Pooling as an Infinitely Strong Prior](#convolution-and-pooling-as-an-infinitely-strong-prior)
  * [Weight Prior](#weight-prior)
- [Variants of the Basic Convolution Function](#variants-of-the-basic-convolution-function)
  * [Effect of Strides](#effect-of-strides)
  * [Effect of Zero Padding](#effect-of-zero-padding)
    + [Zero Padding Strategies](#zero-padding-strategies)
- [Types of Convolution](#types-of-convolution)
  * [Comparing Unshared, Tiled and Traditional Convolutions](#comparing-unshared--tiled-and-traditional-convolutions)
  * [Examples of Unshared, Tiled and Traditional Convolutions](#examples-of-unshared--tiled-and-traditional-convolutions)
    + [Unshared Convolution](#unshared-convolution)
    + [Tiled Convolution](#tiled-convolution)
    + [Traditional Convolution](#traditional-convolution)
    + [Comparing Computation Times](#comparing-computation-times)
- [Structured Outputs](#structured-outputs)
- [Data Types](#data-types)
- [Efficient Convolution Algorithms](#efficient-convolution-algorithms)
  * [Fourier Transform](#fourier-transform)
  * [Separable Kernels](#separable-kernels)
- [Random and Unsupervised Features](#random-and-unsupervised-features)
  * [Greedy Layer-wise Pre-training](#greedy-layer-wise-pre-training)

# Convolutional Neural Networks

Convolutional networks also known as convolutional neural networks, or CNNs, are a specialized kind of neural network for **processing data that has a known grid-like topology.**

# Convolution Operation
The convolution operates on the **input** with a **kernel (weights)** to produce an **output map** given by:<br>
<img src="./images/01.updated_convolution_operation.png"></img>
* **1-D** discrete convolution operation can be given by:<br>
<img src="./images/03.1-D-cnn.png"></img><br>
* **2-D** discrete convolution operation can be given by:<br>
<img src="./images/04.2-D-cnn.png"></img><br>
* **2-D convolution operation** can be visualized as below:<br>
<img src="./images/16.convolution_operation_2.gif"></img><br>
## Example Demonstrating Convolution Operation
<img src="./images/01.convolution_operation.gif"></img>

## Properties of Convolution Operation and Cross-Correlation
### Commutative Property
* Convolution operation is **commutatiive**.
* Commutative property arises because we have **ﬂipped the kernel** relative to the input<br>
<img src="./images/05.commutative_lhs.png"></img>
<img src="./images/06.commutative_rhs.png"></img>

### Cross-Correlation 
* Function which is analogous to convolution operation without flipping the kernel is called **cross-correlation operation.**
* Cross-correlation is **not commutative.**<br>
* **Convolution operation:**<br>
<img src="./images/06.commutative_rhs.png"></img>
* **Correlation operation:**<br>
<img src="./images/07.cross_correlation.png"></img><br>

## Toeplitz Matrix
* **1D** convolution operation can be represented as a **matrix vector product.** 
* The kernel marix is obtained by composing weights into a **Toeplitz matrix.**
* Toeplitz matrix has the property that **values along all diagonals are constant.**<br>
<img src="./images/08.toeplitz_matrix_1d.png"></img><br>

## Block-Circulant and Doubly-Block-Circulant Matrix
* To **extend** the concept of Toeplitz matrix towards **2-D input**, we need to **convert 2-D input to 1-D vector.**
* **Kernel needs to be modified** as before but this time resulting in a **block-circulant matrix.**
* A **circulant matrix** is a special case of a **Toeplitz matrix** where each **row is equal to the row above shifted by one element.**<br>
<img src="./images/09.circulant_matrix.png"></img><br>

* A matrix which is **circulant with respect to its sub-matrices** is called a **block circulant matrix.**<br>
<img src="./images/10.curculant_matrix.png"></img><br>
* If each of the **submatrices is itself circulant**, the matrix is called **doubly block-circulant matrix.**<br>
<img src="./images/11.doubly_circulant_matrix.png"></img><br>

# Motivation
## Sparse Interactions
* In traditional Neural Networks, **every output unit interacts with every input unit.** 
* Convolutional networks, however, typically have **sparse interactions,** by making **kernel smaller than input.**
    * Reduces memory requirements
    * Improves statistical eﬃciency
* In a deep convolutional network, units in the deeper layers may **indirectly interact** with a larger portion of the input.
<img src="./images/13.comparison_sparse_interactions.png"></img><br>


## Parameter Sharing
* Parameter sharing refers to **using same parameter for more than one function in a model.**
* In convolutional neural net, **each member of kernel** is used at **every position of input** i.e. parameters used to compute different output units are **tied together** (all times their values are same).
* **Sparse interactions and parameter sharing combined** can improve eﬃciency of a linear function for **detecting edges** in an image

### Equivariance
* Parameter sharing in a convolutional network **provides equivariance to translation.** <br>
<img src="./images/14.equivariance.png"></img><br>
* Translation of image results in corresponding translation in the output map.
* Convolution operation by itself is **not equivariant to changes in scale or rotation.**<br>
<img src="./images/15.equivariance_rotate_scale.png"></img><br>

# Pooling
* A convolution layer consists of **3 layers -**<br>
     * Convolution
     * Activation (Detector Stage)
     * Pooling
* A pooling function **replaces the output** of net at a certain location with **summary statistic of nearby outputs.**
* Common summary statistics are : **mean, median, weighted average.** <br>
<img src="./images/17.max_pooling.png"></img><br>
* Pooling makes the representation slightly **translation invariant**, in that **small translations** in the input **do not cause large changes in output map.**
* It allows detection of a particular feature **if we only care about its existence**, not its position in an image.
* Pooling **reduces input size to the next layer** in turn reducing the number of computations required upstream.

## Inputs having Variable Size
* **Classification layers** requires **fixed size** of their inputs. 
* **Pooling** makes their **output fixed size** by changing their **pooling size, stride etc.**<br>
<img src="./images/18.variable_sized_inputs.png"></img><br>

## Learned Invariances
* Pooling over feature channels can be used to develop invariance to certain transformations of the input.
* Units in a layer may be **developed to learn rotated features** and then pooled over. This property has been used in **Maxout networks.**<br>
<img src="./images/16.maxout_function.png"></img><br>

# Convolution and Pooling as an Infinitely Strong Prior
## Weight Prior
**Assumptions about weights (before learning)** in terms of acceptable values and range are encoded into the **prior distribution** of weights.<br>


| S.No. | Prior Type        | Variance/Confidence Type                                                                    |
|-------|-------------------|---------------------------------------------------------------------------------------------|
| 1.    | Weak              | High Variance, Low Confidence                                                               |
| 2.    | Strong            | **Narrow range** of values about which we are **confident**  before learning begins.        |
| 3.    | Infinitely strong | Demarkates certain values as **forbidden** completely  assigning them **zero probability.** |


* Convolution imposes an **infinitely strong prior** by making the following **restrictions on weights:**<br>
     * **Adjacent units** must have the **same weight** but shifted in space.
     * Except for a **small spatially connected** region, all **other weights** must be **zero.**
* Features should be **translation invariant.**
* If tasks relies on preserving specific spatial information, then pooling can cause on all features can increase training error.

# Variants of the Basic Convolution Function

In practical implementations of the convolution operation, certain modifications are made which deviate from standard discrete convolution operation -

* In general a convolution layer consists of application of **several different kernels** to the input. Since, convolution with a **single kernel can extract only one kind of feature.**
* The input is generally not real-valued but instead **vector valued.** 
* Multi-channel convolutions are commutative iff **number of output and input channels is the same.**

## Effect of Strides
* **Stride** is the number of **pixels shifts** over the input matrix.
* In order to allow for calculation of features at a **coarser level** strided convolutions can be used. 
* The effect of strided convolution is the same as that of a **convolution followed by a downsampling stage.**
* Strides can be used to **reduce the representation size.**
* Below is an example representing **2-D Convolution, with (3 * 3) Kernel and Stride of 2 units.**<br>
<img src="./images/19.stride_2.gif"></img><br>

## Effect of Zero Padding
* Convolution networks can implicitly zero pad the input V, to make it wider.
* Without zero padding,the width of representation shrinks by one pixel less than the kernel width at each layer.
* Zero padding the input allows to control kernel width and size of output independently.

### Zero Padding Strategies
3 common zero padding strategies are:<br>


| Zero Padding Type       | Properties                                                                                                                                                                                                                                                                                                                                                                                                                                | Example                                                 |
|-------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------|
| **Valid Zero-Padding**  | 1. **No zero padding** is used.<br> 2. Output is computed only at places where **entire kernel lies inside the input.**<br> 3. **Shrinkage > 0**<br> 4. **Limits #convolution layers** to be used in network<br> 5. Input's width = m, Kernel's width = k,<br> **Width of Output = m-k+1**<br>                                                                                                                                                | <img src="./images/20.valid_zer_padding.gif"></img><br> |
| **Same Zero-Padding**   | 1. Just enough **zero padding is added** to keep:<br>      1.a. **Size(Ouput) = Size(Input)**<br> 2. Input is padded by **(k-1) zeros**<br> 3. Since the **#output units connected to border pixels is less** <br> than that for centre pixels, it may **under-represent border pixels.**<br> 4. Can **add as many convolution layers** as hardware can support<br> 5. Input's width = m, Kernel's width = k,<br> **Width of Output = m**<br> | <img src="./images/21.same_zero_padding.gif"></img><br> |
| **Strong Zero-Padding** | 1. The input is padded by enough zeros such that **each input pixel is<br>  connected to same #output units.**<br> 2. Allows us to make an **arbitrarily deep NN.**<br> 3. Can **add as many convolution layers** as hardware can support<br> 4. Input's width = m, Kernel's width = k,<br> **Width of Output = m+k-1**<br>                                                                                                                   | <img src="./images/22.full_sero_padding.gif"></img><br> |


# Types of Convolution

## Comparing Unshared, Tiled and Traditional Convolutions

| Convolution Type            | Properties                                                                                                                                                                                                                                                                                        | Advantages and  Disadvantages                                                                                                                                                                                                                                                                  |
|-----------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Unshared Convolution**    | 1. **No Parameter sharing**.<br> 2. Each output unit performs a **linear operation on its neighbourhood** but parameters are not shared across output units.<br> 3. Captures **local connectivity** while allowing **different features** to be computed at **different spatial locations.**<br>  | **Advantages**<br> 1. **Reducing memory** consumption<br> 2. Increasing **statistical eﬃciency**<br> 3. Reducing the **amount of computation needed** to perform forward and back-propagation.<br>  **Disadvantages**<br> 1. requires **much more parameters** than the convolution operation. |
| **Tiled Convolution**       | 1. Offers a **compromise b/w unshared and traditional convoltion.**<br> 2. Learn a **set of kernels and cycle/rotate them** through space.<br>  3. Makes use of **parameter sharing.**<br>                                                                                                        | **Advantages**<br> 1. Reduces the #parameters in model.                                                                                                                                                                                                                                        |
| **Traditional Convolution** | 1. Equivalent to **tiled convolution with t=1.**<br> 2. Has the **same connectivity as unshared convolution**<br>                                                                                                                                                                                 |                                                                                                                                                                                                                                                                                                |


## Examples of Unshared, Tiled and Traditional Convolutions

### Unshared Convolution
<img src="./images/22.unshared_convolution.png"></img><br>

### Tiled Convolution
<img src="./images/23.tiled_convolution.png"></img><br>

### Traditional Convolution
<img src="./images/24.traditional_convolution.png"></img><br>

### Comparing Computation Times
<img src="./images/25.computation.png"></img><br>

# Structured Outputs
* Convolutional networks can be trained to output **high-dimensional structured output** rather than just a classification score.
* To produce an **output map as same size as input map**, only **same-padded convolutions** can be stacked.
* The output of the first labelling stage can be **refined successively** by another convolutional model. 
* If the models use tied parameters, this gives rise to a type of **recursive model**<br>
<img src="./images/26.structured_outputs.png"></img><br>

| Variable  | Description                                            |
|-----------|--------------------------------------------------------|
| **X**     | Input image tensor                                     |
| **Y**     | Probability distribution over tensor for each pixel    |
| **H**     | Hidden representation                                  |
| **U**     | Tensor of convolution kernels                          |
| **V**     | Tensor of kernels to produce estimation of lables      |
| **W**     | Kernel tensor to convolve over Y to provide input to H |

# Data Types
> The data used with a convolutional network usually consist of several channels,each channel being the observation of a diﬀerent quantity at some point in space or time.<br>

* When output is **variable sized, no extra design change** needs to be made. 
* When output requires **fixed size** (classification), a **pooling stage** with **kernel size proportional to input size** needs to be used.<br>
<img src="./images/27.single_vs_multichanel.png"></img><br>

# Efficient Convolution Algorithms
## Fourier Transform
The Fourier Transform is a tool that breaks a waveform (a function or signal) into an alternate representation, characterized by sine and cosines. <br>
<img src="./images/33.fourier_transform.png" height="200px" width="400px"></img><br>
## Separable Kernels
* Convolution is equivalent to converting both **input and kernel** to frequency domain using a **Fourier transform**, performing **point-wise multiplication of two signals**:<br>
<img src="./images/30.eca_1.png"></img><br>
* Converting back to **time domain** using an **inverse Fourier transform.** <br>
<img src="./images/31.eca_2.png"></img><br>
* When a **d-dimensional kernel** can be expressed as **outer product of d vectors**, one vector per dimension, the kernel is called **separable.**
* The kernel also takes **fewer parameters** to represent as vectors.<br>

| Kernel Type            | Runtime complexity for *d-dimensional kernel* with *w elements wide* |
|------------------------|----------------------------------------------------------------------|
| **Traditional Kernel** | <img src="./images/28.traditional_kernel.png"></img><br>             |
| **Separable Kernel**   | <img src="./images/29.sepearable_kernel.png"></img><br>              |



# Random and Unsupervised Features
To reduce the cost of convolutional network training, we have to use features that are not trained in a supervised way:
* **Random Initialization:** 
     * Layers consisting of **convolution followed by pooling** naturally become **frequency selective and translation invariant** when assigned random weights.
     * **Randomly initialize** several CNN architectures and just **train the last classification layer**.
     * Once a winner is determined, train that model using a **more expensive approach** (supervised approach).
* **Hand Designed Kernels:**
      * Used to **detect edges** at a certain orientation or scale.
* **Unsupervised Training:**
      * Unsupervised pre-training may offer **regularization effect**. 
      * It may also allow for **training of larger CNNs** because of **reduced computation cost.**
      
## Greedy Layer-wise Pre-training
Instead of training an entire convolutional layer at a time, we can **train a model of a small patch:** 
* Train the **ﬁrst layer in isolation.**
* **Extract all features** from the ﬁrst layer **only once.**
* Once the first layer is trained, its **output is stored** and **used as input** for training the next layer.
* We can **train very large models** and incur a **high computational cost** only at inference time.

[Scroll To Top](#convolutional-neural-networks)  











