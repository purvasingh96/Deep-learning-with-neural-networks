# Regularization for Deep Learning

# Overview
* Regularization can be defined as any **modiﬁcation** we make to learning algorithm that is intended to **reduce its [generalization error](https://github.com/purvasingh96/Deep-learning-with-neural-networks/tree/master/Notes/Ch_4_Machine_Learning_Basics#capacity-overfitting-and-underfitting) but not its training error.**
* An eﬀective regularizer is one that makes a proﬁtable trade, **reducing variance** signiﬁcantly while **not overly increasing bias.**
*  Best ﬁtting model is a large model that has been **regularized appropriately.**

# Strategies to make Deep Regularization Model
## Parameter Norm Penalty
* **Purpose :** Limit model's capacity by **adding norm penalty Ω(θ)** parameter to objective function **J**.  
