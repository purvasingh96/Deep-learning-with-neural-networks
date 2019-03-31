# Overview 
Basic idea of an Artifical Neural Network is to mimic a biological neuron with axons, dendrites, nucleus etc.
For a simple neural network, you would need 2 neurons, who pass on information using **Synapse**, flowing from **dendrite** of sending neuron to **terminal axon** of recieving neuron.

### 1. Biological NN v/s Artifical NN
In ANN, the input data will have certain weights attached with them. Before passing data to neuron, the **weighted sum** is calculated and passed on to the **Activation** function.
This function bears a threshold value to check against weighted sum. For e.g if weighted sum is less than threshold value, do not pass the information to next neuron (boolean value 0) else fire a signal (boolean value 1). 

Below diagram shows a basic neural network model :: 

<img src="images/artificial_neural_network_model.PNG" width="600" >

