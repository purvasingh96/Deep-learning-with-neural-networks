# Convolutional Neural Networks

Convolutional networks also known as convolutional neural networks, or CNNs, are a specialized kind of neural network for **processing data that has a known grid-like topology.**

# Convolution Operation
The convolution operates on the **input** with a **kernel (weights)** to produce an **output map** given by:<br>
<img src="./images/01.updated_convolution_operation.png"></img>
* **1-D** discrete convolution operation can be given by:<br>
<img src="./images/03.1-D-cnn.png"></img><br>
* **2-D** discrete convolution operation can be given by:<br>
<img src="./images/04.2-D-cnn.png"></img><br>
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
<img src="./images/07.cross_correlation.png"></img>
