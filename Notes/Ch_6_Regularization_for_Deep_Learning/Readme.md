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
## Parameter Norm Penalty
* **Purpose :** Limit model's capacity by **adding norm penalty Ω(θ)** parameter to objective function **J**.  
