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








