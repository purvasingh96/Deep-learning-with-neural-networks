# Neural Machine Translation (NMT)

## Basic Seq-to-Seq Model

In NMT, we use an encoder and a decoder to translate from one language to another. An encoder decoder architecture looks like this:
<br><br>
It takes in a hidden states and a string of words, such as a single sentence. The encoder takes the inputs one step at a time, collects information for that piece of inputs, then moves it forward. The orange rectangle represents the encoders final hidden states, which tries to capture all the information collected from each input step, before feeding it to the decoder. This final hidden state provides the initial states for the decoder to begin predicting the sequence.

<img src="./images/1. basic seq-to-seq model.png"><img> <br><br>

### Limitation of a basic Seq-to-Seq Model

One major limitation of a basic seq-to-seq model is *information bottle-neck* represented by the figure below:
<img src="./images/2.NMT basic model.png"><img> <br><br>

In case of long sequences of sentences, when the end-user stacks up multiple layers of words, words that are entered at a later stage are given more importance than the words that were entered first.<br><br>
 Because the encoder hidden states is of a fixed size, and longer inputs become *bottlenecked* on their way to the decoder.

Hence, inputs that contain short sentences will work for NMT but long sentences may not work for a basic seq-to-seq model. 
