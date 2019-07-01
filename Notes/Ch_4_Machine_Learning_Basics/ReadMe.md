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

## 
