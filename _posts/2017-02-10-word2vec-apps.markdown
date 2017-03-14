---
layout: post
title:  "Word2Vec on App Descriptions"
date:   2017-03-13 21:43:43 -0600
categories: Machine Learning
---

Sometimes, we come across features which don't may not have any value to the model by themselves but including additional metadata associated with 
such features could be useful and a very good example of such a feature is app ID. The app ID in itself may not be very useful as a feature in a 
model but app descriptions, app category e.t.c could be very valuable depending on what we are modeling. In this post, we map a set of app ID's to
vectors using word2vec. I pulled app descriptions for a sample of ~500 app bundle Ids and the sample can be downloaded [here](https://github.com/p058/word2vec-appdescriptions)


First, we preprocess the app descriptions by removing stopwords, removing punctutation e.t.c


```python

import string
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from gensim.models import word2vec #Force Install numpy(conda install -f numpy) if this scripts hangs when importing this


app_descriptions = pd.read_csv('app_descriptions_sample.csv')
StopWords = set(stopwords.words("english"))

def tokenize(text):
    tokens = [x.strip() for x in text.split(' ')]
    tokens = [x.lower() for x in tokens if x.isalpha() and len(x) > 2 and x not in StopWords]
    return tokens


def preprocess_text(x):
    x_lower = x.lower()
    x_nopunc = x_lower.translate(None, string.punctuation)
    return x_nopunc
    
app_descriptions['Description'] = app_descriptions['Description'].apply(lambda x : preprocess_text(x))
app_descriptions['Words'] = app_descriptions['Description'].apply(lambda x : tokenize(x))

```

Now, lets train word2vec model on the app descriptions

```python

sentences = list(app_descriptions.Words)
num_features = 200  # Size of vector
min_word_count = 1  # minimum frequency to be included for training. (=1 to train on all words)
context = 10  # Context window size (number of words to left/right to be used as context)
downsampling = 1e-2 # proportion to down sample frequently seen words

model = word2vec.Word2Vec(sentences, \
                              size=num_features, min_count=min_word_count, \
                              window=context, sample=downsampling, iter=50)

##All words                           
words = model.vocab.keys()

##word2vec dict
vectors = {}
for word in words:
    vectors[word] = model[word]
    

```

Now we have a model that maps words to vector but we want to map app Id's to vector and each app Id has a set of words in its description, 
so we have few options to do the mapping.
a) take an average of all vectors corresponding to word list for an app
b) append the vectors corresponding to all words for an app (the sizes of vectors could vary for each app so you may have to
 limit the number of dimensions in this case)
c) take a weighted average of the word vectors where the weights are calculated as tf-idf scores for the words. (For ex: 
if App1 --> (word1, word2, word3)
and word1 --> vec1, word2 --> vec2, word3 --> vec3
App1 --> vec1*tf1+vec2*tf2+vec3*tf3
)

Features extracted using weighted average seems to have more semantic meaning.
 
Lets get tf-idf scores for all the words

```python
from collections import Counter

dataWords = app_descriptions.Words.apply(lambda x: list(set(x)))
idf = Counter([j for sublist in dataWords for j in sublist])
idfLog = [(j[0], np.log(len(app_descriptions) / float(j[1]))) for j in idf.items()]
idfLogDict = dict(idfLog)

app_descriptions['tf'] = app_descriptions.Words.apply(lambda x: [(j[0], j[1] / float(len(x))) for j in Counter(x).items()])

app_descriptions['tf-idf'] = app_descriptions['tf'].apply(
    lambda x: sorted([(j[0], j[1] * idfLogDict[j[0]]) for j in x], key=lambda x: x[1], reverse=True))

```

Now lets convert app ID's to vectors

```python

def tf_word(x, vectors, num_features = num_features):
    """
    this function takes in the tf-idf scores/word2vec mappings for all words corresponding to an app and returns a weighted average
    of the  app descriptions
    
    """
    res = np.ones(num_features)

    for word, tf in x:
        res += vectors[word] * tf

    return np.array(res) / float(1 + len(x))

app_descriptions.index = app_descriptions['App Bundle Id']
apptfidf_dict = app_descriptions['tf-idf'].to_dict()

app_vector = {}

for app in apptfidf_dict.keys():
    app_vector[app] = tf_word(apptfidf_dict[app], vectors)


```

Lets see how good the trained app vectors are:

```python
a = app_vector[datingapp] #dating app
b = app_vector[weatherapp1] #weather app

from sklearn.metrics.pairwise import cosine_similarity
print cosine_similarity(a.reshape(1,-1), b.reshape(1,-1))[0,0])

##0.40323566

a = app_vector[weatherapp1] #weather app
b = app_vector[weatherapp2] #weather app

print cosine_similarity(a.reshape(1,-1), b.reshape(1,-1))[0,0])

##0.97410025
```

Code for pulling app descriptions can be found [here](https://github.com/p058/word2vec-appdescriptions)


