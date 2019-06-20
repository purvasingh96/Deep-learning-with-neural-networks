# Linear Algebra

Table of Contents
=====================
* [Scalars, Vectors, Matrices and Tensors](Scalars,-Vectors,-Matrices-and-Tensors)
  * [Scalars](Scalars)
  * [Vectors](Vectors)
  * [Matrices](Matrices)
    * [Properties of Matrices and Vectors](Properties-of-Matrices-and-Vectors)
* [Matrix Decomposition](Matrix-Decomposition)
  * [Eigen Decomposition](Eigen-Decomposition)
  * [Single Value Decomposition](Single-Value-Decomposition)
  * [Moore-Penrose Pseudoinverse](Moore-Penrose-Pseudoinverse)
* [Trace Operator](Trace-Operator)
  

# Scalars, Vectors, Matrices and Tensors

## Scalars
* single number 
* write scalar in *italics*
* E.g. *Let s ∈ R be the slope of the line*

## Vectors 
* Array of numbers
* Identifying points in space, with each element giving the coordinate along a different axis.

## Matrices
* 2-D Array of numbers
* uppercase, bold-face e.g. **A**

### Properties of Matrices and Vectors

| S.No | Matrix/Vector Property                     | Description                                                                           |
|------|-------------------------------------|---------------------------------------------------------------------------------------|
| 1.   | **Transpose of Matrix**                 |<img src="images/transpose.png" width="270" height="60" >                                                                                       |
| 2.   | **Matrix Addition**                     |<img src="images/Selection_096.png" width="200" height="40" >                                                                                       |
| 3.   | **Matrix Multiplication**               |<img src="images/Selection_097.png" width="200" height="60" >                                                                                       |
| 4.   | **Transpose of Matrix Product**         |<img src="images/Selection_098.png" width="200" height="60" >                                                                                       |
| 5.   | **Identity Matrix**                     |<img src="images/identity_matrix.png" width="200" height="60" >                                                                                       |
| 6.   | **Inverse Matrix**                      |<img src="images/inverse_matrix.png" width="200" height="60" >                                                                                       |
| 7.   | **Linear Combination of vectors**       |<img src="images/linear_combination'.png" width="200" height="60" >                                                                                       |
| 8.   | **Span of set of vectors**              |<img src="images/span.png" width="150" height="60" >                                                                                       |
| 9.   | **Linearly Indipendent set of vectors** | If **no vector is linear combination** of other vectors in the set;                        |
| 10.  | **Square Matrix**                       | Equal no. of rows and columns                                         |
| 11.  | **Left/Right Inverse**                  | For square matrix, **left inverse = right inverse**                                       |
| 12.  | **Norms**                               | Determines **size of vector**;<br><img src="images/norms.png" width="200" height="60" ><img src="images/normas_2.png" width="200" height="60" >                                                             |
| 13.  | **Properties of Norms (function *f*)**                 |<img src="images/properties_of_norm.png" width="350" height="150" >                                                                                       |
| 14.  | **Euclidean Norm**                      |<img src="images/euclidean_norm.png" width="200" height="50" >                                                                                       |
| 15.  | **L1 Norm**                             | Determines diff. between elements that are exactly 0 and elements that are close to 0;<br><img src="images/l1_norm.png" width="200" height="50" > |
| 16.  | **Max Norm**                            |<img src="images/max_norm.png" width="200" height="50" >                                                                                       |
| 17.  | **Frobenius Norm**                     | Measures **size of matrix**. (Analogus to euclidean-norm)                                 |
| 18.  | **Dot product of vectors (x, y)**       |<img src="images/dot_product.png" width="200" height="50" ><img src="images/dot_prod_@.png" width="250" height="50" >                                                                                       |
| 19.  | **Diagnol Matrix**                      |<img src="images/diagnol_matrix.png" width="100" height="45" ><img src="images/diag_matrix_2.png" width="150" height="50" >                                                                                       |
| 20.  | **Identity Matrix**                     | Matrix whose diagnol enteries are 1                                                   |
| 21.  | **Symmetric Matrix**                    |<img src="images/symmetric_matrix.png" width="200" height="50" >                                                                                       |
| 22.  | **Unit Vector**                         |<img src="images/unit_vector.png" width="200" height="50" >                                                                                       |
| 23.  | **Orthogonal Vectors**                  |<img src="images/orthogonal_matrix.png" width="200" height="50" >                                                                                       |
| 24.  | **Orthonormal Vectors**                 | **Orthogonal** vectors with **unit norm**;<br><img src="images/orthonormal_matrix.png" width="200" height="50" >                                                     |
| 25.  | **Orthogonal Matrix**                   | Square matrix with rows and columns **orthonormal respectively**.                         |


# Matrix Decomposition
## Eigen Decomposition
**Eigendecomposition** is a way of **breaking/decomposing** matrix into smaller matrices (analogus to *prime factorization*).<br>
<img src="images/eigen_decompost.png" width="200" height="50" ><br>
**Eigen Vector* is a non-zero vector ***v***, which upon being multiplied by matrix **A**, alters only the scale of ***v***.<br>
<img src="images/eigen_vector.png" width="200" height="50" ><br>
Below figure shows before and after multiplying *eigen vector* with *eigen value* :<br>
<img src="images/before_after_ev.png" width="400" height="200" ><br>


* Matrix whose all eigen values are -
  * positive is called **positive definite**
  * positive or zero-valued is called **positive semi-definite**
  * negative is called **negative definite**
  * negative or zero-valued is called **negative semi-definite**

## Single Value Decomposition
SVD factorizes matrix into singular values and singular vectors. In SVD, matrix **A** can be decomposed as follows - <br>
<img src="images/SVD.png" width="200" height="50" >
### Properties - 
* **A** - (m, n) matrix
* **U** - (m, m) orthogonal matrix, columns of **U** are called **left-singular vectors**
* **D** - (m, n) diagnol matrix, not necessarily square, elements along diagnol **D** are called **singular values of A** 
* **V** - (n, n) orthogonal matrix, columns of **V** are called **right-singular vectors**

## Moore-Penrose Pseudoinverse
Usually matrix inversion is not possible for non-square matrices. To solve below linear equation, in case **A is non-square matrix** , we use Moore-Penrose pseudoinverse formula to **find solution to x** - <br>
<img src="images/mpr.png" width="150" height="50" ><br>
<img src="images/mpr_formula.png" width="250" height="65" ><br>
Here, **U, D, V** are SVD of **A**.<br><br>
*Pseudo-inverse of **D** is obtained by* - 
* take reciprocal of non-zero elements 
* take transpose of resultant matrix

# Trace Operator
* Gives **sum of diagnol enetries** of matrix.<br>
<img src="images/trace.png" width="250" height="65">

* **Frobenius norm** can be re-written in terms of Trace operator as follows -
<img src="images/frob_in_trace.png" width="250" height="65">

* **Properties of Trace operator** -
  1. <img src="images/trace_comm.png" width="200" height="60"><br>
  2. <img src="images/trace_transpose.png" width="200" height="60">
