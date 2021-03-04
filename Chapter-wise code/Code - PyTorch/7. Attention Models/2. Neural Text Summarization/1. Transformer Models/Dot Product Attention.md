# Dot Product Attention

Below steps describe in detail as to how a *dot-product attention* works:

*Imp: Queries: German words.*

1. Let's consider the phrase in English, *"I am happy"*. 
First, the word *I* is embedded, to obtain a vector representation that holds continuous values which is unique for every single word.<br><br>
<img src="../images/7.step - 1.png" width="50%"></img><br>

2. By feeding three distinct linear layers, you get three different vectors for queries, keys and values.<br><br>
<img src="../images/8. step - 2.png" width="50%"></img><br>

3. Then you can do the same for the word *am* to output a second vector. <br><br>
<img src="../images/9. step - 3.png" width="50%"></img><br>

4. Finally the word *happy* to get a third vector and form the queries, keys and values matrix.<br><br>
<img src="../images/10. step - 4.png" width="50%"></img><br>

5. From both the Q matrix and the K matrix, the attention model calculates weights or scores representing the relative importance of the keys for a specific query.
<img src="../images/11. step - 5.png" width="50%"></img><br>

6. These attention weights can be understood as alignment scores as they come from a dot product. <br><br>
<img src="../images/12. step - 6.png" width="50%"></img><br>

7. Additionally, to turn these weights into probabilities, a softmax function is required.<br><br>
<img src="../images/13. step - 7.png" width="50%"></img><br>

7. Finally, multiplying these probabilities with the values, you will then get a weighted sequence, which is the attention results itself.<br><br>
<img src="../images/14. step - 8.png" width="50%"></img><br>


