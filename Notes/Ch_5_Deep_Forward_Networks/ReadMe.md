# Deep Forward Networks

# Overview
* **Deep forward networks** also called **Multi Layer Perceptron (MLPs)** are deep learning models whose aim is to approximate some function *f* *.
* **MLP is different from classifier** as follows -
  * For **Classifier**, y=f*(x), maps input **x** to category **y**.
  * For **MLP**, y=f(x;θ) learns parameter θ that results in **best function approximation**.
  
* **They are called feed-forward because of the following information flow-**<br>
  `information ==> x ==> intermediate computations defining f ==> final o/p y   `
* They are called **networks** because they are **computations** of many functions.
* **Depth and Width of Network**
  * Given 3 functions, <img src="./images/01.function_composition.png" width="130px;" height="30px;"></img> connected in chain to form <img src="./images/02.function_chain_rule.png"  width="180px;" height="30px;"></img>, then <img src="./images/03.first_layer.png" height="30px"></img> is called **first layer.**
  * Overall **length of chain** gives **depth** of model.
  * **Final layer** is called **Output Layer.**
  
 * **Hidden Layers and Model-Width**
    * Aim is to match f(x) with f*(x)
    * **Training examples** specify each output layer to produce value close to <img src="./images/04.hidden_layer.png" height="30px;"></img>
    * Learning algorithm makes use of above layer to **best implement approximation of f*.**
    * Since, training data **doesnot show desired o/p for each of these layers,** they are called **hidden layers.**
    * **Width of model = Dimensionality of Hidden layers.**
  
  * **Non-Linear Transformations**
    * Linear model can be extended to **non-linear functions of x** by applying linear model directly to transformed input **φ(x)**.
    * Mapping **φ** can be chosen as follows:
       * **Very generic φ** : Enough capacity, poor generalization.
       * **Manually engineered φ** : Dominant approach, takes too much time.
       * **Deep-learning of φ** : Requires learning of φ; Highly generic; Human designer finds right "general" function instead of right function. 
       
# Rectified Linear Unit (ReLU) Activation Function 
## Problem Statement
Our model provides a **functiony=f(x;θ)**, and **learning algorithm** will adapt **θ** to make **f similar to XOR function y =f\*(x).**

* **MSE** loss function **J(θ) -**<br>
<img src="./images/06.mse_linear_transformations.png"></img>
* Linear model's **definition**<br>
<img src="./images/07.linear_model.png"></img>
* Minimizing **J(θ)** w.r.t **w** and **b** gives **w=0 and b=0.5** which is wrong.

## Linear Model Approach
* Constructing a **linear feed-forward network as below** <br>
<img src="./images/09.linear_deep_network.png"></img>
* Final **complete model:** <img src="./images/10.complete_linear_xor.png" height="40px" width="250px"></img>
* Making <img src="./images/03.first_layer.png" height="30px"></img> as linear would make **entire feed-forward network as linear.**
* Assuming **linear approach** and let <img src="./images/08.linear_hidden.png" height="35px"></img> and <img src="./images/10.linear_output.png" height="35px"></img>, in that case  <img src="./images/11.linear_network.png" height="35px"></img>, which needs **non-linear functions to describe features.**

## Non-Linear Model Approach
* Most neural networks use **non-linear function** to describe features by using **affine transformation** controlled by **learned parameters**,followed by a ﬁxed **nonlinear function** called an **activation function**.
* Affine transformation from **x to h** is defined and **activation function g** defined as:<br> <img src="./images/13.relu_applied.png"></img>
* Recommended activation function is **ReLU**, defined as:<br><img src="./images/14.relu.png"></img><br>
<img src="./images/15.relu_graph.png" height="300px" width="350px"></img>
* Final **non-linear model** would be:<br>
<img src="./images/16.non_linear_model.png"></img>

# Gradient Based Learning
To apply gradient-based learning we must choose a **cost function**, and we must choose how to represent **output of the model**.

## Cost Functions
* Cost functions for neural networks is **approximately same as linear functions.** 
* Cost function used is **cross-entropy between training data and model's prediction.**<br>
<img src="./images/17.cross_entropy.png"></img><br>
* Advantage of using maximum-likelihood for cost function is that it removes burden for designing **cost functions for each model.**
* **Gradient** of network should be **large and predictable.**
* **Saturatable functions** make **activation function small** which produces the model's output (exponent functions that saturate when their argument is negative). Solution is to use **negative logarithmic functions.**

### Conditional Statistics
* Instead of learning a **full probability distribution p(y | x;θ),** we could learn just one **conditional statistic of y given x**.
* Making cost fucntion as being **functional** rather than just **function**.
* Solving an optimization problem w.r.t function requires a mathematical tool called **calculus of variations.**
   * **Optimization Problem**<br>
   <img src="./images/18.optimization_problem.png"></img><br>
   * **First result derived using calculus of variations** - predicts **mean** of y for each value of x<br>
   <img src="./images/19.first_result_calculus_tools.png"></img><br>
   * **Second result derived using calculus of variations** - predicts **median** of y for each value of x, also known as **mean absolute error.**<br>
   <img src="./images/20.second_result_calculus_tools.png"></img><br>
   * Mean squared error and mean absolute error lead to **poor results when used with gradient-based optimization.**








