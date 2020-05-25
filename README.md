# Exploring deep learning neural networks

This repository is just a collection of notebooks that I'm using to explore various deep learning (DL) concepts and APIs (such as keras).  The mathematics behind the networks is straightforward but tedious so I'm focusing on applying networks.

Previously I tried DL for regression on some tabular data, but it struggles to keep up with the simple random forest. DL shines with unstructured data so I turned to the standard hello world of MNIST classification:

* [MNIST with a vanilla network](notebooks/mnist-vanilla.ipynb)
* [MNIST using convolutional network](notebooks/mnist-CNN.ipynb)

The next step is to play around with tokenizing text and representing words and documents with sparse work vectors:

* [Text tokenization, word indexing, word vectors](notebooks/word-vectors.ipynb)
* [Dense but random word vectors](notebooks/dense-random-embeddings.ipynb)

Before jumping into dense word vectors, I wanted to explore the whole embedding mechanism using some simple categorical variables. Before using the built-in embedding layer with keras, I wanted to build some examples manually.
 
* [Manual movie embeddings via movie+user->rating](notebooks/catvar-embeddings-homebrew.ipynb) (Got ok but not great results here.)
* [Separate layers for movie and user IDs joined into a DL pipeline](notebooks/catvar-embeddings-split-homebrew.ipynb) (Got good results here.)
* [Keras embeddings via movie+user->rating](notebooks/catvar-embeddings-keras.ipynb) (slightly older than the previous two now.)

* [Word embeddings using Keras](notebooks/word-embeddings-keras.ipynb)

Recurrent neural networks

Generating text

PyTorch

* [A start on collaborative filtering](notebooks/collaborative-filtering.ipynb)
