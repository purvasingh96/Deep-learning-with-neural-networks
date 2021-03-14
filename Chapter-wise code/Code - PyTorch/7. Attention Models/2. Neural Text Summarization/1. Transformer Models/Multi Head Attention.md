# Multi-Head Attention

## Overview

1. Input to multi-head attention is a set of 3 values: Queries, Keys and Values.<br><br>
<img src="../images/26. step -1 .png" width="50%"></img><br>
2. To achieve the multiple lookups, you first use a fully-connected, dense linear layer on each query, key, and value. This layer will create the representations for parallel attention heads. <br><br>
<img src="../images/27. step - 2.png" width="50%"></img><br>
3. Here, you split these vectors into number of heads and perform attention on them as each head was different.<br><br>
4. Then the result of the attention will be concatenated back together.<br><br>
<img src="../images/28. step - 3.png" width="50%"></img><br>
5. Finally, the concatenated attention will be put through a final fully connected layer.<br><br>
<img src="../images/29. step - 4.png" width="50%"></img><br>
6. The scale dot-product is the one used in the dot-product attention model except by the scale factor, one over square root of DK. DK is the key inquiry dimension. It's normalization prevents the gradients from the function to be extremely small when large values of D sub K are used.<br><br>
<img src="../images/30. step - 5.png" width="50%"></img><br>

## Summary
<img src="../images/31. multi-head attention.png"></img><br>

## Maths behind Multi-Head Attention
