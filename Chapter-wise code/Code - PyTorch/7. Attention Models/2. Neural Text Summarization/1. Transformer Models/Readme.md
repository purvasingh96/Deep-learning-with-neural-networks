# Transformer Models

## What was wrong with Seq2Seq Models?

1. No parallel computaions. For longer sequence of text, a seq2seq model will take more number of timesteps to complete 
the translation and as we know, with large sequences, the information tends to get lost in the network (vanishing gradient).
LSTMs and GRUs can help to overcome the vanishing gradient problem, but even those will fail to process long sequences.<br><br>
<img src="../images/1. drawbacks of seq2seq.png" width="50%"></img><br>

2. In a conventional Encoder-decoder architeture, the model would again take T timesteps to compute the translation.<br><br>
<img src="../images/2. basic encoder-decoder.png" width="50%"></img><br>
  
## Transformers - Basics
```buildoutcfg
TLDR:
1. In RNNs, parallel computing is difficult to implement.
2. For long sequences in RNN, there is loss of information.
3. RNNs face the problem of vanishing gradient.
4. Transformer architecture is the solution.
```

1. Transformers are based on attention and don't require any sequential computation per layer, only one single step is needed.
2. Additionally, the gradient steps that need to be taken from the last output to the first input in a transformer is just one.
3. Transformers don't suffer from vanishing gradients problems that are related to the length of the sequences.<br><br>
<img src="../images/3. transformer model.png" width="50%"></img><br>
4. Transformer differs from sequence to sequence by using multi-head attention layers instead of recurrent layers.<br><br>
<img src="../images/4. multi-head attention.png" width="50%"></img><br>

5. Transformers also use positional encoding to capture sequential information. The positional encoding out puts values to be added to the embeddings. That's where every input word that is given to the model you have some of the information about it's order and the position.
<img src="../images/5. positional encoding.png" width="50%"></img><br>

6. Unlike the recurrent layer, the multi-head attention layer computes the outputs of each inputs in the sequence independently then it allows us to parallelize the computation. But it fails to model the sequential information for a given sequence. That is why you need to incorporate the positional encoding stage into the transformer model.
