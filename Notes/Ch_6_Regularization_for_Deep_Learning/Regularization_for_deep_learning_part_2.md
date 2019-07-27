# Regularization for Deep Learning : Part -2


# Parameter Tying
## Paramter Dependency
* L2 regularization/weight decay **penalizes models parameters from deviating from fixed value of 0.**
* Sometimes we need **other ways to express prior knowledge of parameters.**
* We may know from domain and model architecture that ther should be **some dependencies between model parameters.**

## Motivation and Goal
We want to express that certain parameters **should be close** to one another.

## Scenario of Parameter Tying
* Considering a situation, where 2 models are performing **same classification task, with somewhat different input distributions**.<br>
* Parameters of one model are regularized (trained in **supervised paradigm**) using norm penalty, **to be close to parameters** of another model (trained in **unsupervised paradigm**).<br>
* <img src="./images/45.param_tying_01.png"></img>
* Two models will map the input to **2 different, but related output**<br>
<img src="./images/46.param_tying_02.png"></img>

## L2 Penalty for Parameter Tying
* If tasks are similar enough, then **params of both models should be close to each other** <br>
<img src="./images/47.param_tying_03.png"></img>
* We can leverage this information via **regularization, using parameter norm penalty**
<img src="./images/48.param_tying_04.png"></img>

# Parameter Sharing

* Another approach to make parameters of different model close to one another is to **force sets of parameters to be equal**. This method of regularization is called **Parameter Sharing.**<br>
* We interpret the various models or model components **as sharing a unique set of parameters.**<br>
* Only **subset of params need to be stored in memory.**
* Example: Paramter sharing in CNNs<br>
<img src="./images/44.param_sharing_for_CNN.png"></img>

