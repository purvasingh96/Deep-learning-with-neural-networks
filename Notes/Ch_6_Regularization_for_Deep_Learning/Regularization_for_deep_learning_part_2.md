# Regularization for Deep Learning : Part -2

# Parameter Sharing

* Considering a situation, where 2 models are performing **same classification task, with somewhat different input distributions**.<br>
* Parameters of one model are regularized (trained in **supervised paradigm**) using norm penalty, **to be close to parameters** of another model (trained in **unsupervised paradigm**).<br>
* Another approach to make parameters of different model close to one another is to **force sets of parameters to be equal**. This method of regularization is called **Parameter Sharing.**<br>
* We interpret the various models or model components **as sharing a unique set of parameters.**<br>
* Only **subset of params need to be stored in memory.**
* Example: Paramter sharing in CNNs<br>
<img src="./images/44.param_sharing_for_CNN.png"></img>
