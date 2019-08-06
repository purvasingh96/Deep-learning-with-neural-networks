# Optimization for Training Deep Models

# Learning v/s Optimizing
* Machine learning acts **indirectly** by trying to **optimize performance measure, P,** defined w.r.t test set.
* Goal of learning is to reduce **expected generalization errror.**
* Learning algorithms **reduce cost functions**
    * By **minimizing expected loss** on training data set.
    * In the hope that **indirect optimization will improve performance.**
* Expectation is taken across **data-generating distribution**, rather than finite training set.<br>
<img src='./images/01.cost_function_for_entire_data_set.png'></img>

## Emperical Risk
* Simplest way to **convert ML problem to an optimization problem** is to **minimize expected loss** on the training set.
* Replacing **true distribution p(x, y)** with **empirical distribution ˆp(x, y)** deﬁned by the training set.<br>
<img src='./images/02.emperical_risk.png'></img>
* **m** is number of training examples.
* Training process based on **minimizing average training error** is known as **empirical risk minimization**.
* Rather than **optimizing risk directly, optimize empirical risk** and hope that **risk decreases signiﬁcantly** as well.

## Disadvantages of Emperical Risk Method
* Prone to **over-fitting.**
* Models with **high capacity** can **memorize** training data.
* Most effective loss functions are based on **SGD**, but losses like **0-1 loss have no useful derivatives.**

# Surrogate Loss Function and Early Stopping
* In situations where loss function is **difficult to/cannot be optimized,** we optimize **surrogate loss function**, acting as proxy with several advantages -
    * Differentiable
    * Improves robustness.
## Difference between Genral Optimization and Optimizing Training Algorithms
*  **Training algorithms** do not usually halt at a local minimum. ML algorithm **minimizes surrogate loss function** and halts when **convergence criterion based on early stopping** is satisﬁed.
* Training **halts** while the surrogate loss function **still has large derivatives.**

# Batch and Mini-batch Algorithms
* **Objective function** usually **decomposes as a sum** over training examples.
* **Optimization** in ML, typcally computes loss and updates parameters **iteratively**, e.g. **stochastic gradient descent.**
* Computing **exact expectation**/ **using entire training set** (batch/deterministic algorithm) can be **computationally expensive.**
* Minibatch algorithms compute expectation by **randomly sampling small number of examples** from dataset, then taking the average over only those examples.

## Various Information extracted from Batch/Minibatch Algorithms
* Second order methods **that use Hessian matrix,** require much **larger batch sizes.**
* Minibatches should be **sampled randomly.**
* **Minibatch SGD** follows gradient of **true generalization error**, as long as **no examples are repeated**.
* When using an **extremely large training set**, **underﬁtting** and **computational eﬃciency** becomes the predominant concerns.

## Batch v/s Minibatch
* Each iteration in **minibatch** may have **poor optimization performance** than batch algorithm.
* However, after **many iterations**,mini-batch algorithm generally **converges to optimal state.**

