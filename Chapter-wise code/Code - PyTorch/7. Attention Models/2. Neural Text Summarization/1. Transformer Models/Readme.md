# Transformer Models

## What was wrong with Seq2Seq Models?

1. No parallel computaions. For longer sequence of text, a seq2seq model will take more number of timesteps to complete 
the translation and as we know, with large sequences, the information tends to get lost in the network (vanishing gradient).
LSTMs and GRUs can help to overcome the vanishing gradient problem, but even those will fail to process long sequences.<br><br>
<img src="../images/1. drawbacks of seq2seq.png" width="50%"></img><br>

2.  