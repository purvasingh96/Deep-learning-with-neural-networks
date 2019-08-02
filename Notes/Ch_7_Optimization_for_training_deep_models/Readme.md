# Optimization for Training Deep Models

# Learning v/s Optimizing
* Machine learning acts **indirectly** by trying to **optimize performance measure, P,** defined w.r.t test set.
* Goal of learning is to reduce **expected generalization errror.**
* Learning algorithms **reduce cost functions**
    * By **minimizing expected loss** on training data set.
    * In the hope that **indirect optimization will improve performance.**
* Expectation is taken across **data-generating distribution**, rather than finite training set.<br>
<img src='./images/01.cost_function_for_entire_data_set.png'></img>
