# Causal (self) Attention

## Overview

1. In Causal attention, keys and words are from the same sentence. Hence the name, *self attention*.
2. Queries in causal attention are allowed to look at words that occured in the past.<br>
<img src="../images/21. causal attention overview.png" width="50%"></img><br><br>

## Causal Attention Math
The difference between a dot-product attention and causal attention is the matrix *Mask*.

1. For causal attention, you could compute attention weights in the same way as before `softmax(Q.KTranspose)`. 
But that way, you are allowing the model to attend to words in the future.<br>
<img src="../images/22. step - 1.png" width="50%"></img><br><br>

2. To solve this issue, you add a mask by a sum of size L by L. 
So you compute softmax of Q times K transpose + M. 

<img src="../images/23. step - 2.png" width="50%"></img><br><br>

3. When you add M to Q times K transpose, all values on the diagonal and below which correspond to queries attending words in the past are untouched. 
All other values become minus infinity. After a softmax, all minus infinities will become equal to 0, as exponents of negative infinity is equal to 0, so it prevents words from attending to the future.
<img src="../images/24. step - 3.png" width="50%"></img><br><br>

4. The last step is the same as dot-product attention!

 <img src="../images/25. step - 4.png" width="50%"></img><br><br>
 
 ## Next Up
 Next, we will learn about multi-head attention. You can find the read-me file for the same [here].