# Back Propagation and Other Differentiation Algorithms

# Overview
* A **feedforward neural network** accepts **input x** and produces **output y**, information **ﬂows forward** through the network. The input **x** provides initial information which propagates through **hidden units and ﬁnally produces y**. This is called **forward propagation**.<br>
* **Back-propagation algorithm (backprop)**, allows the information from the **cost to then ﬂow backward through network** in order to **compute gradient**.

* Numerically evaluating analytical expression for **gradient** can be very expensive.

## Computational Graphs

* Computational graphs are used to describe **back-propagation algorithms more precisely.**
* Each **node** describes a **variable**
* Graphs are accompanied by **operations**, which is a simple **function of one or more variables.**
* If a **variable y** is computed by applying an **operation to a variable x**, then we draw a **directed edge from x to y**.<br>
<img src="./images/01.computational_graphs.png"></img>

## Chain Rules of Calculus
* Suppose that **x ∈ Rm**, **y ∈ Rn** ,g **maps from Rm to Rn**, and f **maps from Rn to R**. 
* If **y=g(x) and z=f(y)** then ***chain rule of calculus states that***-<br>
<img src="./images/02.chain_rul.png"></img>
* Equivalent **vector notion can be stated as -**<br>
<img src="./images/03.vectorized_chain_rule.png"></img><br>
<img src="./images/04.jacobian_matrix.png"></img><br>

* **Gradient of a variable x** can be obtained by multiplying **Jacobian matrix ∂y/∂x by a gradient ∇yz**.
* Back-propagation algorithm consists of performing **Jacobian-gradient product for each operation in the graph.**


## Recursively Applying the Chain Rule to Obtain Backprop
* Any procedure that **computes the gradient** will need to choose whether to **store sub expressions or to recompute them several times**.<br>
<img src="./images/05.graphs_and_chain_rule.png"></img><br>
* Eqution (1) suggests an implementation in which computes **f(w) only once and store it in the variable x**. Used when **memory is low.**
* Equation(2) **re-computes value of f(w)**  each time it is needed. Used when **memory is limited.**
* **Amount of computation** required for performing back-propagation **scales linearly with #edges in G**. 
     * Computation for **each edge** corresponds to computing a **partial derivative** (of one node with respect to one of its parents) as well as **performing one multiplication and one addition.**

* Back-propagation algorithm is designed to **reduce number of common subexpressions** without regard to memory.
* Backprop performs on the order of **one Jacobian product per node in the graph.**


