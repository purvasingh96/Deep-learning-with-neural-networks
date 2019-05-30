# [Introduction to Deep Learning](http://d2l.ai/chapter_introduction/intro.html) 


Table of Contents
=================
* [Definition](#Definition)   
* [Key Components](#Key-Components)
    * [Data](#Data)
    * [Model](#Model)
    * [Loss](#Loss)
    * [Algorithm](#Algorithm)
* [Types of Machine Learning](#Types-of-Machine-Learning)
    * [Supervised Machine Learning](#Supervised-Machine-Learning)
      * [Regression](#Regression)
      * [Classification](#Classification)
      * [Tagging](#Tagging)
      * [Sequential Learning](#Sequential-Learning)

## Definition
 As ML algorithm accumulates more experience, typically in the form of observational data or interactions with an environment, their performance improves.In deep learning, the learning is the process by which we discover the right setting of the knobs coercing the desired behaviour from our model. Below is the flow of a typical training process - <be>
 <img src="../images/training_process.png" width="550" >
 
 ## Key Components
 Following are the core components, irrespective of ML problem we are trying to solve - 
 * **Data**<br>
 >1. Workable data is given a numerical representation with *features* and *covariates* as its numerical attributes.
 >2. Numerical values can be referred as *Vectors* and this fixed-length of vectors is called *dimensionality of data.*
 >3. *More and right* data means more powerful model.
 * **Model**
 >1. *A model* would ingest data and output predictions for the same.
 * **Loss**
 >1. *Objective functions* are the standards to measure how good or bad our model is.
 >2. *Lower* the objective function, the better, hence these functions are named as *Loss/Cost functions.* 
 >3. Most common objective function is *squared error.* 
 * **Algorithm**
 >1. Once the model is ready, we *optimize* it to minimize the objective function.
 >2. The most popular optimization algorithms for neural networks follow an approach called *gradient descent*. 
 
 ## Types of Machine Learning
 ## Supervised Machine Learning
 >1. Predicts **targets** for given input of data.
 >2. Targets are called **labels** and denoted by **y**.
 >3. Input data points are called **instances** and denoted by **x**.
 >4. The goal is to produce a model  **𝑓𝜃**  that maps an input  **𝑥**  to a prediction  **𝑓𝜃(𝑥)** (**the learned model**).
 
 ## Types of Supervised Learning Problems
 ### Regression
 >1. When our targets (**y**) take on arbitrary real values in some range, we call this a regression problem.
 >2. The goal of Regression problem is to **predictions** that closely resemble **target values**.
 >3. A good rule of thumb is that any **How much? or How many?** problem should suggest **regression**.
 >4. We try to **reduce distance** between **predicted values** and actual **target values**, which is called *Loss*.
 >5. Most common type of losses are **L1 Loss occuring due to Laplace noise** and **L2 Loss occuring due to Gaussian noise**
 
 ### Classification
 >1. In classification, we look at a feature vector and then predict which category (formally called **classes**), among some set of options, an example belongs. 
 >2. When we have more than two possible classes, we call the problem **multiclass classification**. 
 >3. The common loss function for classification problems is called **cross-entropy**.
 
 ### Tagging
>1. The problem of learning to predict classes that are not mutually exclusive is called **multi-label classification**. 
>2. **Auto-tagging** problems are typically best described as multi-label classification problems.

### Sequential Learning
>1. Sequential learning comes into picture when input data might **not have same dimensions** and are **not mutually exclusive**.
>2. It requires to either **ingest sequence of input or emit sequence of output or both**.
>3. Example of sequential learning are -
 >   * **language translation**.
 >* **Automatic speech recognition**
 >* **Text to Speech**
 >* **Machine Translation**

 ## Unspuervised Learning
 Unsupervised learning is a type of machine learning algorithm used to draw inferences from datasets consisting of input data without labeled responses.  Following are the type of unsupervised learning methods -
 >1. **Clustering** - Clustering is used for exploratory data analysis to find hidden patterns or grouping in data.
 >2. **Principal Component Analysis** - PCA a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables (entities each of which takes on various numerical values) into a set of values of linearly uncorrelated variables called principal components.
 >3. **Representational Learning** - It is used to describe entities and their relations.
 
 ## Reinforcement Learning
 At each time step  𝑡 , the agent receives some observation  𝑜𝑡  from the environment, and must choose an action  𝑎𝑡  which is then transmitted back to the environment. Finally, the agent receives a reward  𝑟𝑡  from the environment.<br>
 The agent can replay random past experiences to help it perform action and make decision.
