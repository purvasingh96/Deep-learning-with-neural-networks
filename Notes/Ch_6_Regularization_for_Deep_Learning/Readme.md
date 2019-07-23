# Regularization for Deep Learning

# Overview
* Regularization can be defined as any **modiﬁcation** we make to learning algorithm that is intended to **reduce its [generalization error](https://github.com/purvasingh96/Deep-learning-with-neural-networks/tree/master/Notes/Ch_4_Machine_Learning_Basics#capacity-overfitting-and-underfitting) but not its training error.**
* Best ﬁtting model is a large model that has been **regularized appropriately.**
* Goal of regularization is to **prevent overfitting** by imposing some **strategies** such as -
    * Put extra constraint on ML model.
    * Add extra term in objective function.
    * Impose ensemble method.
 * Error due to **bias -** Difference between the **expected (or average) prediction** of our model and **correct value** which we are trying to predict. 
 * Error due to **variance -** Variance is how much the **predictions for a given point vary** between different realizations of the model
 * **Bias** has a **negative first-order derivative** in response to model complexity while **variance** has a **positive slope.**
 
 # Bias Variance Trade-Off
 * An eﬀective regularizer is one that makes a proﬁtable trade, **reducing variance** signiﬁcantly while **not overly increasing bias.**<br>
 <img src="./images/01.bias_variance_trade_off.png"></img>

# Strategies to make Deep Regularization Model
# Parameter Norm Penalty
* Limits the model's capacity by **adding norm penalty Ω(θ)** parameter to objective function **J**. <br>
 <img src="./images/02.norm_penalities.png"></img>
* Does not modify the model in inference phase, but **adds penalties in learning phase**.
* Norm penalty **penalizes only weights**, leaving **biases unregularized.** 
* Also known as **Weight Decay.**

## Modified Objective Function
* ***w*** denotes all the weights that should be aﬀected by a norm penalty, **vector θ** denotes all the parameters, including both ***w*** and the **unregularized parameters.**
* Regularized objective function **decreases both J and θ.**
* Setting **α ∈[0, ∞)** to 0 results in **no regularization** and **larger values of α** corresponds to **more regularization**.

# L2 Parameter Regularization

* Commonly known as **Weight decay**, this regularization strategy drives **weights closer to origin.** by adding regularization term :<br>
<img src="./images/03.l2_norm_penalty.png"></img>

## L2 norm calculation
* Substituting **squared l2 norm** as penalty -<br>
<img src="./images/04.l2_norm_substitution.png"></img>

* Calculating **gradient** -<br>
<img src="./images/05.l2_weight_update.png"></img>

* Applying **weight update -**<br>
<img src="./images/06.l2_weight_update_2.png"></img><br>
<img src="./images/07.l2_weight_update_3.png"></img>

## Effect of L2 Norm Paramterization
* Making **quadratic approximation to objective function,** in the neighborhood of value of weights that obtains **minimal unregularized training cost,** w*<br>
<img src="./images/08.l2_minimum_training_cost.png"></img>
* **Quadratic approximation** of J gives <br>
<img src="./images/09.l2_quadratic_approximation.png"></img><br>
   * Here **H** refers to **positive sem-definite Hessian Matrix** of J w.r.t **w evaluated at** w* 
   * Minimum of J^ occurs when -<br>
   <img src="./images/10.l2_approximaton_minimal.png"></img><br>





