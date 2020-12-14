# Machine Translation using Basic Linear Algebra

## Algorithm

### Generate French and English Word Embedding

Here, we have 3 given parameters:
1. `en_embeddings`: English words and their corresponding embeddings.<br>
  <img src="./images/en_embeddings.png"></img>

2. `fr_embeddings`: French words and their corresponding embeddings.<br>
  <img src="./images/fr_embedding.png"></img>
 
3. `en_fr`: English to French dictionary.<br>
  <img src="./images/en_fr_train.png"></img>

Now, we have to create an English embedding matrix and French embedding matrix:<br>
  <img src="./images/en_fr_embeddings.png.png"></img><br>
```python
# loop through all english, french word pairs in the english french dictionary
    for en_word, fr_word in en_fr.items():

        # check that the french word has an embedding and that the english word has an embedding
        if fr_word in french_set and en_word in english_set:

            # get the english embedding
            en_vec = english_vecs[en_word]

            # get the french embedding
            fr_vec = french_vecs[fr_word]

            # add the english embedding to the list
            X_l.append(en_vec)

            # add the french embedding to the list
            Y_l.append(fr_vec)

    # stack the vectors of X_l into a matrix X
    X = np.vstack(X_l)

    # stack the vectors of Y_l into a matrix Y
    Y = np.vstack(Y_l)
```
