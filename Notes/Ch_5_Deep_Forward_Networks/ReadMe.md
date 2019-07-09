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
  
