# [Introduction to Deep Learning](http://d2l.ai/chapter_introduction/intro.html) 

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
 
