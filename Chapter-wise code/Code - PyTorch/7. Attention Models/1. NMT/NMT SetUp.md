# Setup for Machine Translation

## Data in NMT

Below we have the data sequence in English, *I'm hungry*, and on the right, the corresponding German equivalent. 
Further down we have, *I watch the soccer game*, and, the corresponding German equivalent. 
We are going to have a great many of these inputs. One thing to note here is that the data set used is not entirely clean.

<img src="./images/10. data in NMT.png" width="50%"></img><br><br>

## Pre-requisites

1. *Input*: Take English sentence as input.
2. *Tokenization*: State-of-the-art models use pre-trained word vectors, else, represent words with one-hot vectors to create the input.
3. *Padding*: Pad the tokenized sequence to make the inputs of equal length.<br><br>
<img src="./images/11. NMT setup-english.png" width="50%"></img><br><br>
4. Repeat steps 1-3 for the German sentences as well.<br><br>
<img src="./images/12. NMT setup - german.png" width="50%"></img><br><br>
5. Keep track of index mappings with word2index and index2word mappings.
5. Use start-of-sentence `<SOS>` and end-of-sentence `<EOS>` tokens to represent the same.

### Teacher Forcing

Let us assume we want to train an image captioning model, and the ground truth caption for an  image is “Two people reading a book”. Our model makes a mistake in predicting the 2nd word and we have “Two” and “birds” for the 1st and 2nd prediction respectively.
1. *Without Teacher Forcing*, we would feed “birds” back to our RNN to predict the 3rd word. Let’s say the 3rd prediction is “flying”. Even though it makes sense for our model to predict “flying” given the input is “birds”, it is different from the ground truth.
<br><img src="./images/13. No teacher forcing.png"></img><br>
2. *With Teacher Forcing*, we would feed “people” to our RNN for the 3rd prediction, after computing and recording the loss for the 2nd prediction.
<br><img src="./images/14. with teacher forcing.png"></img><br>

## Training NMT

1. The initial `select` makes two copies. Each of the input tokens represented by zero (English words) and the target tokens (German words) represented by one.<br>
 <img src="./images/15. step - 1.png" width="30%"></img><br><br>
 
2. One copy of the input tokens are fed into the inputs encoder to be transformed into the key and value vectors. <br>
<img src="./images/16. step - 2.png" width="30%"></img><br><br>

3. While a copy of the target tokens goes into the pre-attention decoder.<br>
<img src="./images/17. step - 3.png" width="30%"></img><br><br>

4. The pre-attention decoder is transforming the prediction targets into a different vector space called the query vector. That's going to calculate the relative weights to give each input weight. The pre-attention decoder takes the target tokens and shifts them one place to the right. This is where the teacher forcing takes place. Every token will be shifted one place to the right, and in start of a sentence token, will be a sign to the beginning of each sequence.<br> 
<img src="./images/18. step - 4.png" width="30%"></img><br><br>

5. Next, the inputs and targets are converted to embeddings or initial representations of the words.<br>
<img src="./images/19. step - 5.png" width="30%"></img><br><br>

6. Now that you have your query key and value vectors, you can prepare them for the attention layer. The mask is used after the computation of the Q, K transpose. This before computing the softmax, the where operator in your programming assignment will convert the zero-padding tokens to negative one billion, which will become approximately zero when computing the softmax. That's how padding works.<br>
<img src="./images/20. step - 6.png" width="30%"></img><br><br>

7. The residual block adds the queries generated in the pre-attention decoder to the results of the attention layer. <br>
<img src="./images/21. step - 7.png" width="30%"></img><br><br>

8. The attention layer then outputs its activations along with the mask that was created earlier. <br>
<img src="./images/22. step - 8.png" width="30%"></img><br><br>

9. It's time to drop the mask before running everything through the decoder, which is what the second Select is doing. It takes the activations from the attention layer or the zero, and the second copy of the target tokens, or the two. Would you remember from way back at the beginning. These are the true targets which the decoder needs to compare against the predictions. <br>
<img src="./images/23. step - 9.png" width="30%"></img><br><br>

10. Then run everything through a dense layer or a simple linear layer with your targets vocab size. This gives your output the right size.<br>
<img src="./images/24. step - 10.png" width="30%"></img><br><br>

11.  Finally, you will take the outputs and run it through LogSoftmax, which is what transforms the attention weights to a distribution between zero and one. <br>
<img src="./images/25. step - 11.png" width="30%"></img><br><br>

12. Those last four steps comprise your decoder.<br>
<img src="./images/26. step - 12.png" width="30%"></img><br><br>

13. The true target tokens are still hanging out here, and we'll pass down along with the log probabilities to be matched against the predictions.<br>
<img src="./images/27. step - 13.png" width="30%"></img><br><br>



 