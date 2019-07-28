# Regularization for Deep Learning : Part - 2

- [Regularization for Deep Learning : Part - 2](#regularization-for-deep-learning---part---2)
- [Parameter Tying](#parameter-tying)
  * [Paramter Dependency](#paramter-dependency)
  * [Motivation and Goal](#motivation-and-goal)
  * [Scenario of Parameter Tying](#scenario-of-parameter-tying)
  * [L2 Penalty for Parameter Tying](#l2-penalty-for-parameter-tying)
- [Parameter Sharing](#parameter-sharing)
- [Bagging and Ensemble Methods](#bagging-and-ensemble-methods)
  * [How does Bagging work?](#how-does-bagging-work-?)
  * [Bagging Example](#bagging-example)
  * [Usage of Bagging](#usage-of-bagging)
  * [Tacit Rules of Bagging](#tacit-rules-of-bagging)

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

# Bagging and Ensemble Methods
* Bagging (**bootstrap aggregating**) - technique for reducing general-ization error by **combining several models.**
* Train several diﬀerent **models separately**, then have **all models vote** on the output for test examples. This strategy is known as **model averaging.** Techniques employing this strategy are known as **ensemble methods.**
<img src="./images/49.bagging.png"></img><br>

## How does Bagging work?
* Considering **k regression models** (with minimizing MSE).
* Suppose each model makes  error **εi** on each example, then<br>
<img src="./images/50.bagging_2.png"></img><br>
* Error made by **average prediction of models is**<br>
<img src="./images/51.bagging_3.png"></img><br>
* **Expected square error** of ensemble predictor is <br>
<img src="./images/52.bagging_4.png"></img><br>
  * If errors are **perfectly correlated** and **c=v**, **MSE=v** and model won't work at all.
  * If errors are **perfectly uncorrelated** and **c= 0**, error will be only **v/k**.
  
## Bagging Example 
* Below is an example of **8-detector**, where **first ensemble** learns that a **loop at top** implies the digit is 8 and **second ensemble** learns that **loop at bottom** implies 8.<br>
<img src="./images/52.eight_detector.png"></img><br>

## Usage of Bagging
* **Trend regression** on the data ozone-temperature.
* **Gray line** is regression line with each samples.
* **Red line** is average line<br>
<img src="./images/53.bagging_examples.png"></img><br>

## Tacit Rules of Bagging
* **OOB (Out-Of-bag) Sampling**
  * Special rule for sampling with **replacement.**
  * If we sample the example with **random sampling replacement**, selecting probability of each example is:
<img src="./images/54.OOB_1.png"></img><br> 
  * If **N is large enough**, then<br>
<img src="./images/55.OOB_2.png"></img><br> 

* Bagging in **neural networks**
  * Random initialization
  * Random selection  of minibatches
  * Differences in hyperparameter

# Dropout
## Overfitting
* Excessive focus on training data can lead to **poor results on actual test data**.<br>
<img src="./images/56.overfitting.png"></img><br> 

## Solution for Overfitting
* Regularization
  * L1-norm penalty
  * L2-norm penalty
* Data augmentation
* Dropout (2012)
   * A method of bagging applied to neural networks
   * An inexpensive but powerful method of regularizing a broad family of models
   
## What is Dropout?
* A technique that **omits a portion of network.**
* Trains the ensembles consisting of all subnetworks that can be formed by **removing nonoutput units** from an underlying base network.
* A unit from network can be removed by multiplying **its output value by zero**.
* Typically, **input unit** is included with **probability 0.8**, and a **hidden unit** is included with **probability 0.5**.
<img src="./images/58.dropout_modelling_00'.png"></img>

## Effects of Dropout
* Avoids **co-adaptation**, a trend that some neurons tend to represent similar features
* Captures **clear features** by avoiding co-adaptations.
<img src="./images/59.effect_of_dropout.png"></img>

## Problems Addressed by Dropout
* **Omitting of neurons**, to mimic the voting in ensemble technique, instead of building the multiple models.
* **Product the probability** that a neurons will survive to weight, at **inference level** (**accumulation of votes** from all models).

## Differences between Bagging and Dropout
| S.No. | Bagging                                                                                                              | Dropout                                                                                                                                                                                                                     |
|-------|----------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1.    | All models are **independent**                                                                                           | Models **share parameters**.<br> Each model **inherits** a different subset of parameters<br> **from parent neural network.**                                                                                                           |
| 2.    | Each model is **trained to** <br> **convergence** on its own training<br> set.                                               | Most models are **not explicitly trained** at all.<br> Tiny **fraction of possible subnetworks** are each trained<br> for single step and **parameter sharing** causes **remaining subnetworks**<br> to arrive at **good settings of network**. |
| 3.    | **Prediction** of ensemble is given by <br> **arithmetic mean of probability distribution** <img src="./images/57.prob_dist.png"></img><br> generated by **each model *i*.** as follows -<br> <img src="./images/58.am_pd_bagging.png"></img><br>| Each submodel defined by **mask vector µ** defines a **probability**<br> **distribution.** Arithmetic mean over **all masks** is given by<br><img src="./images/59.am_pd_dropout.png"></img><br>                                                                                      |






