# Machine Translation using Basic Linear Algebra

## 1. Generate French and English Word Embedding

Here, we have 3 given parameters:
1. `en_embeddings`: English words and their corresponding embeddings.<br>
  <img src="./images/en_embeddings.png"></img>

2. `fr_embeddings`: French words and their corresponding embeddings.<br>
  <img src="./images/fr_embedding.png"></img>
 
3. `en_fr`: English to French dictionary.<br>
  <img src="./images/en_fr_train.png"></img>

Now, we have to create an English embedding matrix and French embedding matrix:<br>
  <img src="./images/en_fr_embeddings.png"></img><br>

## 2. Linear Transformation of Word Embeddings
Given dictionaries of English and French word embeddings we will create a transformation matrix `R`. In other words, given an english word embedding, `e`, we need to multiply `e` with `R`, i.e., (`eR`) to generate a new word embedding `f`.

### Describing Translation Problem as the Minimization Problem
We can describe our translation problem as finding a matrix `R` that minimizes the following equation:<br> 
<img src="./images/translation_problem.png"></img><br>
For this, we calculate the loss by modifying the original *Forbenius norm* :<br>
Original Forbenius Norm: <img src="./images/original_forbenius_norm.png"></img><br>
Modified Forbenius Norm: <img src="./images/modified_forbenius_norm.png"></img><br>
Finally, our loss funtion will look something like this:<br>
<img src="./images/final_loss_function.png"></img><br>
<img src="./images/description.png"></img>




