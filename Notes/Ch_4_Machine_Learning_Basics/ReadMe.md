# Machine Learning Basics 
Machine learning isessentially a form of applied statistics with increased emphasis on the use ofcomputers to statistically estimate complicated functions and a decreased emphasison proving conﬁdence intervals around these functions

# Learning Algorithms
A computer program is said to learn from **experience E** with respect to some class of **tasks T** and **performance measure P**, if its **performance** at tasks in T, as measured by P, improves with experience E.<br>

| Term                   | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
|------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Task, T**                | ML tasks are described in terms of how **ML system should process an example**.<br> **Example** is collection of **features** , quantitatively measured from event that ML model will process.<br> 1. **Classification**-which of the *k* categories, input belongs to?<br> 2. **Classification with missing inputs** - ML model defines *single function mapping* from vector to categorical input.<br> 3. **Regression** - Predict output, given some input.   |
| **Perfomance Pressure, P** | 1. Measure *accuracy, error-rate* of model.<br> 2. Evaluate P, using *test-set* (completely separate from *train-set*).<br>                                                                                                                                                                                                                                                                                                                                   |
| **Experience, E**          | 1. Based on E, ML algorithms are classified as *unsupervised* and *supervised* learning algos.                                                                                                                                                                                                                                                                                                                                                                |

## Linear Regression
* **Goal :** Build a system that takes vector **x є R**, and predict value of scalar **y є R** as output. Output can be defined as follows, where **b** is the intercept term -<br>
<img src="images/06. linear_regression.png" width="170" height="50" ><br>
* Term **b**, is often called **bias parameter,** in the sense that o/p of transformation is *biased towards being b*, in absence of any input. 
* **w є R** is set of **vectpr parameters/weights** that determines how each feature affects prediction.
* Model's performance can be measured on the basis of **Mean Squared Error (MSE)** on *test-set*.<br>
<img src="images/02.MSE.png" width="250" height="60" ><br>
* Improve ML model, by improving **w** such that it **reduces *MSE(test)***, by observing training set, <img src="images/03. training_set.png" width="150" height="30" ><br> 
* To get **optimum weight, w value**, minimize MSE(train), i.e. solve for where its **gradient = 0** (system of **Normal Equations.**).<br>.
<img src="images/04. MSE-train.png" width="150" height="30" ><br>
<img src="images/05. Optimum_weight.png" width="300" height="60" ><br>

# Capacity, Overfitting and Underfitting
* **Generalization -** Ability to perform on previously unobserved inputs.
* **Generalization/Test-Error** - Expected value of error on new input.
* To improve model's performance, reduce **test-error.** Test error can be defined as follows -<br><img src="images/07. test_error.png" width="200" height="60" ><br>
* Set of **i.i.d assumptions** describes data-generating process with probability distribution over single sample.
* **i.i.d** assumptions refers to examples in each data-set are **indipendent** and training and test data are **identically distributed.**
* Expected training error of random model equals expected testing error of that model.
* Factors that determine how well ML model will perform are -<br>
    * smaller training error
    * smaller gap between training and testing error
* **Underfitting** - model's inability to achieve smaller training error.
* **Overfitting** - gap between training and testing error is too large.
* **Capacity** - model's capacity is its ability to ﬁt a wide variety of functions.
* **Hypothesis space** - set of functions that the learning algorithm is allowed to select as being the solution.
* ML algos perform best when their capacity is appropriate for the true complexity of the task.
* **Representational Capacity** - ( choice of model + hypothetical space)
* **Effective Capacity <= Representational Capacity**<br>
<img src="images/08. Error_graph.png" width="400" height="200" ><br>

* **Baeys Error** - difference between true probability distribution and true probability distribution.
* **No Free Lunch Theorem** - ML aims to find rules that are **probably correct** about **most** members of the set rather than finding rules that are surely correct about certain set of members.<br>
Theorem states that averaged over all possible data sets, **no ML algorithm is universally any better than other**.

* **Weight Decay -** ML algos are given preference in **hypothesis space.** We can transform linear regression to include weight decay. The criterion *J(**w**)*, has preference for weights with smaller **L2 norm**<br>
<img src="images/09. weight_decay.png" width="200" height="50" ><br>
   * λ is a value chosen ahead of time that controls the strength of our preference for smaller weights.
   * λ= 0, we impose no preference
   * larger λ forces weights to become smaller.
   * In general, *we can regularize a model that learns a **functionf(x;θ)** by adding a penalty called **regularizer** to cost function.*
   
* **Regularization** is any modiﬁcation we make to learning algorithm that is intended to **reduce its generalization error** but not training error.

# Hyperparameters and Validation Sets

*  **Hyper-parameters -** 
      * Settings related to ML algos that can be used to control algorithm’s behavior.
      * Sometimes settings is chosen as hyper-paramter, because it is **not appropriate to learn that hyperparameter**               (controlling model's capacity) on the training set, since they would always **chose maximum possible model                   capacity**,resulting in **overfitting**.
     
     
* **Validation-set -**
      * To solve the above problem of overfitting due to chosing of maximum model's capacity, we need set of examples that model has not observed. This set is called *validation-set (constructed from training data).*
      
# Estimators, Bias, Variance
## Estimators 

### Point Estimator 
* Point estimation is the attempt to provide the single **best** prediction of some quantity of interest.
* Point estimation for θ is represented by ˆθ.<br>
<img src="images/10. point_estimator.png" width="200" height="40" ><br>
* A good estimator is a function, whose output is close to true value θ that generated training data.

### Function Estimator
<img src="images/11. function_estimator.png" width="200" height="40" ><br>
* In function estimation,approximate f with a model or estimateˆf.
* Function estimator **ˆf** is simply a point estimator in **function space**.

## Bias
* Bias of an estimator is defined as -<br>
<img src="images/12. bias.png" width="200" height="40" ><br>
* Estimator ˆθm is said to be unbiased if bias(ˆθm) =0
* An estimatorˆθmis said to be **asymptotically unbiased** if<br>
<img src="images/13. asymptotically_unbiased.png" width="200" height="40" ><br>
* **Gaussian and Bernoulli distribution are unbiased.**
* **Variance of Gaussian distribution** is biased with bias of<br> <img src="images/13. bias_of_variance_of_gaussian_distribution.png" width="100" height="40" ><br>

## Standard Variance









