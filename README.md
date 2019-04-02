# Overview 
Basic idea of an Artifical Neural Network is to mimic a biological neuron with axons, dendrites, nucleus etc.
For a simple neural network, you would need 2 neurons, who pass on information using **Synapse**, flowing from **dendrite** of sending neuron to **terminal axon** of recieving neuron.

## 1. Biological NN v/s Artifical NN
In ANN, the input data will have certain weights attached with them. Before passing data to neuron, the **weighted sum** is calculated and passed on to the **Activation** function.
This function bears a threshold value to check against weighted sum. For e.g if weighted sum is less than threshold value, do not pass the information to next neuron (boolean value 0) else fire a signal (boolean value 1). 

Below diagram shows a basic neural network model :: 

<img src="images/artificial_neural_network_model.PNG" width="600" >

## 2. Installing TensorFlow in Ubuntu
1. TensorFlow is supported for Ubuntu versions 16.0.4 or later. If you are using an older version (14..0.4), [upgrade your Ubuntu to a higher version](https://wiki.ubuntu.com/XenialXerus/ReleaseNotes).
2. Following are the **pip instructions** to download tensorflow -


        $ sudo apt-get install python3-pip python3-dev
        # Ubuntu/Linux 64-bit, CPU only, Python 3.5
        $ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.9.0-cp35-cp35m-linux_x86_64.whl
        $ sudo pip3 install --upgrade $TF_BINARY_URL`

