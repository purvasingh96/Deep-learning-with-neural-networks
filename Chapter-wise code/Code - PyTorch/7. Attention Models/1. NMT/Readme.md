# Neural Machine Translation (NMT)

## Basic Seq-to-Seq Model

In NMT, we use an encoder and a decoder to translate from one language to another. An encoder decoder architecture looks like this:
<br><br>
It takes in a hidden states and a string of words, such as a single sentence. The encoder takes the inputs one step at a time, collects information for that piece of inputs, then moves it forward. The orange rectangle represents the encoders final hidden states, which tries to capture all the information collected from each input step, before feeding it to the decoder. This final hidden state provides the initial states for the decoder to begin predicting the sequence.

<img src="./images/1. basic seq-to-seq model.png" width="50%"><img> <br><br>

### Limitation of a basic Seq-to-Seq Model

One major limitation of a basic seq-to-seq model is *information bottle-neck* represented by the figure below:
<img src="./images/2.NMT basic model.png" width="60%"><img> <br><br>

In case of long sequences of sentences, when the end-user stacks up multiple layers of words, words that are entered at a later stage are given more importance than the words that were entered first.<br><br>
 Because the encoder hidden states is of a fixed size, and longer inputs become *bottlenecked* on their way to the decoder.

Hence, inputs that contain short sentences will work for NMT but long sentences may not work for a basic seq-to-seq model.

## Word Alignment

Word Alignment is the task of finding the correspondence between source and target words in a pair of sentences that are translations of each other.<br>
<img src="./images/3. word alignment.png" width="50%"><img> <br><br>
When performing word alignment, your model needs to be able to identify relationships among the words in order to make accurate predictions in case the words are out of order or not exact translations.

In a model that has a vector for each input, there needs to be a way to focus more attention in the right places. Many languages don't translate exactly into another language. To be able to align the words correctly, you need to add a layer to help the decoder understand which inputs are more important for each prediction.<br>
<img src="./images/4. alignment and attention.png" width="70%"><img> <br><br>

## Attention and Alignment
Attention is basically an additional layer that lets a model focus on what's important. 
Below is a step-by-step algorithm for NMT:
1. *Prepare the encoder hidden state and decoder hidden state.*
2. *Score each of the encoder hidden state by getting its dot product between each encoder state and decoder hidden states.*<br>
    2.1. *If one of the scores is higher than the others, it means that this hidden state will have more influence than the others on the output.*
3. *Then you will run scores through softmax, so each score is transformed to a number between 0 and 1, this gives you your attention distribution.*
4. *Take each encoder hidden state, and multiply it by its softmax score, which is a number between 0 and 1, this results in the alignments vector.*
5. *Now just add up everything in the alignments vector to arrive at what's called the context vector, which is then fed to the decoder.*

<img src="./images/5. Calculating alignment for NMT model.png"><img> <br><br>

## Information Retreival via Attention

TLDR: Attention takes in a query, selecting a place where the highest likelihood to look for the key, then finding the key.

1. The attention mechanism uses encoded representations of both the input or the encoder hidden states and the outputs or the decoder hidden states.

2. The keys and values are pairs. Both of dimension N, where N is the input sequence length and comes from the encoder hidden states. 

3. The queries come from the decoder hidden states.

4. Both the key value pair and the query enter the attention layer from their places on opposite ends of the model.

5. Once inside, the dot product of the query and the key is calculated (measure of similarity b/w key and query). Keep in mind that the dot product of similar vectors tends to have a higher value.

6. The weighted sum given to each value is determined by the probability (run through softmax function) that the key matches the query.

7. Then, the query is mapped to the next key value pair and so on and so forth. This is called *scaled dot product attention*.<br><br>

<img src="./images/6. Inside attention layer.png" width="50%"></img><br><br>

## Attention Visualization

Consider a matrix, where words of one query (Q) are represented by rows  keys (K) and words of the keys are represented by values (V).
<br>

The value score (V) is assigned based on the closeness of the match.<br>

```buildoutcfg
Attention = Softmax(QK^T)V
``` 
<br><br>
<img src="./images/7. attention visual - 1.png" width="40%"></img> <img src="./images/8. NMT with attention.png" width="60%"></img> <br><br>

### Flexible Attention

In a situation (as shown below) where the grammar of foreign language requires a difference word order than the other, the attention is flexible enough to find the connection. <br><br>

<img src="./images/9. flexible attention.png" width="50%"></img><br><br>

The first four tokens, the agreements on the, are pretty straightforward, but then the grammatical structure between French and English changes. Now instead of looking at the corresponding fifth token to translate the French word zone, the attention knows to look further down at the eighth token, which corresponds to the English word area, glorious and necessary.  








 


