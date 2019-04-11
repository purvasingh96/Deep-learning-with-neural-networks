import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import pickle
import random
from collections import Counter
# for more details on nltk :: https://github.com/purvasingh96/Natural-Language-Processing/blob/master/README.md


"""
This code (specifically function sample_handling) can be explained with the help of a simple example:

let our lexicon (dictionary of words) be : [coke, pepsi, fanta, miranda]

Let our contents from file we read be ::

I like pepsi. Initially I used to like fanta. I am a huge fan of coke but I don't like its packaging bottle.......

Step1:  Our initial feature-set would be a zeros array of size same as lexicon. -> [0, 0, 0, 0]
        Once we start reading lines, we see [pepsi] -> so now index of pepsi in feature-set array is 1,
        Hence we update our feature-set array to : [0, 1, 0 ,0]

Step2:  Next, we encounter fanta -> [0, 1, 1, 0]
        Next, we encounter coke  -> [1, 1, 1, 0]
                                                                            |--> [1, 0] specify pos and [0,1] specify neg classification
End of iteration and final feature-set would look like :: [                 |
                                                            [1, 1, 1, 0], [1, 0],
                                                            [1, 0, 20, 1], [0,1],
                                                            ......
                                                        ]
                                                

"""

lemmatizer = WordNetLemmatizer()
no_of_lines = 10000

# creating a dictonary
# after tokenizing every sentence

def create_lexicon(pos, neg):
    lexicon = []
    with open(pos, 'r') as f:
        contents = f.readlines()
        for l in contents[:no_of_lines]:
            all_words = word_tokenize(l)
            lexicon += list(all_words)

    # with open(neg, 'r') as f:
    #     contents = f.readlines()
    #     for l in contents[:no_of_lines]:
    #         all_words = word_tokenize(l)
    #         lexicon += list(all_words)

    # removing meaningless / redundant words by applying lemmatizing
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    # word_count = {'car': 6356, 'the':7878, ....}
    word_count = Counter(lexicon)
    least_common_words = []
    for w in word_count:
        # least common words
        # to improve our model
        if 1000> word_count[w]>50:
            least_common_words.append(w)


    return least_common_words

def sample_handling(sample, lexicon, classification):
    featureset = []

    with open(sample, 'r') as f:
        contents = f.readlines()
        for line in contents[:no_of_lines]:
            current_words = word_tokenize(line.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]

            # >>> np.zeros(5)
            # array([ 0.,  0.,  0.,  0.,  0.])

            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1
                    featureset.append([features, classification])

    return featureset


def create_features_sets_and_labels(pos, neg, test_size=0.1):
    lexicon = create_lexicon(pos, neg)
    features = []
    features += create_features_sets_and_labels('pos.txt', lexicon, [1,0])
    features += create_features_sets_and_labels('neg.txt', lexicon, [0,1])
    random.shuffle(features)

    testing_size = int(test_size*(len(features)))

    # x is features, y is labels

    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])

    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])

    return train_x, train_y, test_x, test_y


if __name__ == '__main__' :
    train_x, train_y, test_x, test_y = create_features_sets_and_labels('pos.txt', 'neg.txt')
    with open('sentiment_set.pickle', 'wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)




