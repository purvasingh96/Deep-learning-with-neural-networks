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

### Attention and Alignment
Below is a step-by-step algorithm for NMT:
1. *Prepare the encoder hidden state and decoder hidden state.*
2. *Score each of the encoder hidden state by getting its dot product between each encoder state and decoder hidden states.*<br>
    2.1. *If one of the scores is higher than the others, it means that this hidden state will have more influence than the others on the output.*
3. *Then you will run scores through softmax, so each score is transformed to a number between 0 and 1, this gives you your attention distribution.*
4. *Take each encoder hidden state, and multiply it by its softmax score, which is a number between 0 and 1, this results in the alignments vector.*
5. *Now just add up everything in the alignments vector to arrive at what's called the context vector, which is then fed to the decoder.*

<img src="./images/5. Calculating alignment for NMT model.png"><img> <br><br>
