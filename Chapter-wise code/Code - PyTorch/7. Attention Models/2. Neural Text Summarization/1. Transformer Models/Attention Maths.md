# Attention Maths

```buildoutcfg
Q: Queries (embeddings of German words)
K: Keys (embeddings of English words)
V: Values
D: Dimensionality of word embeddings
Lq: no. of. queries
Lk: no. of keys

```

1. Input to attention: Q, K, V Often Vs are same as Ks.

2. dim[Q] = [Lq, D];
3. dim[K] = [Lk, D];
4. dim[V] = [Lk, D]

<br>
<img src="../images/15. step - 1.png" width="50%"></img> <br><br>

4. A query Q, will assign each key K, a probability  that key K is a match for Q. Similarity is measured by taking dot
product of vectors. So Q and K are similar iff `Q dot K` is large. <br>
<img src="../images/16. step - 2.png" width="50%"></img> <br><br>

5. To make attention more focused on best matching keys, use softmax `(softmax(Q.KTranspose))`. Hence, we now calculate a matrix of Q-K probabailities
often called *attention weights*. The shape of this matrix is `[Lq, Lk]`.<br>

6. In the final step, we take values and get weighted sum of values, weighting each value Vi by the probability that the key Ki matches the query.<br>
<img src="../images/17. step - 3.png" width="50%"></img> <br><br>

7. Finally the attention mechanism calculates the dynamic or alignment weights representing the relative importance of the inputs in this sequence.<br>
<img src="../images/18. step - 4.png" width="50%"></img> <br><br>

8. Multiplying alignment weights with input sequence (values), will then weight the sequence. A single context vector can then be calculated using the sum of weighted vectors.<br>
<img src="../images/19. step - 5.png" width="50%"></img> <br><br>
