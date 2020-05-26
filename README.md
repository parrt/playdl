# Exploring deep learning neural networks

This repository is just a collection of notebooks that I'm using to explore various deep learning (DL) concepts and APIs (such as keras).  The mathematics behind the networks is straightforward but tedious so I'm focusing on applying networks.

## Deep learning image classification

Previously I tried DL for regression on some tabular data, but it struggles to keep up with the simple random forest. DL shines with unstructured data so I turned to the standard hello world of MNIST classification:

* [MNIST with a vanilla network](notebooks/mnist-vanilla.ipynb)
* [MNIST using convolutional network](notebooks/mnist-CNN.ipynb)

## Text processing

The next step is to play around with tokenizing text and representing words and documents with sparse work vectors:

* [Text tokenization, word indexing, word vectors](notebooks/word-vectors.ipynb)
* [Dense but random word vectors](notebooks/dense-random-embeddings.ipynb)

## Categorical variable embedding

Before jumping into dense word vectors, I wanted to explore the whole embedding mechanism using some simple categorical variables. Before using the built-in embedding layer with keras, I wanted to build some examples manually.
 
* [Manual movie embeddings via movie+user->rating](notebooks/catvar-embeddings-homebrew.ipynb) (Got ok but not great results here.)
* [Separate layers for movie and user IDs joined into a DL pipeline](notebooks/catvar-embeddings-split-homebrew.ipynb) (Got good results here.)
* [Keras embeddings via movie+user->rating](notebooks/catvar-embeddings-keras.ipynb) (slightly older than the previous two now.)

* [Word embeddings using Keras](notebooks/word-embeddings-keras.ipynb)

## PyTorch

PyTorch seems a bit more low-level but it's very cool and easy to use. It is essentially a numpy with automatic differentiation package, but also includes a `torch.nn` layer that looks very much like Keras. First, I went through some exercises following their tutorial to get use to the library and how to use the auto differentiation:

* [Gradient descent in pytorch](notebooks/pytorch-gradient-descent.ipynb)

Now, I've had some very good luck building my own collaborative filtering mechanism using my own gradient descent with momentum in raw pytorch. I am more or less able to get the same predictive ability using matrix factorization that I did with the deep learning version above. Implemented gradient descent with momentum and then full AdaGrad.  Didn't do any minibatching.

* [Collaborative filtering with pytorch](notebooks/collaborative-filtering.ipynb) (Cool. learned to display interactive graphs with [bokeh](https://docs.bokeh.org/en/latest/index.html) too!)<br>
<img src="images/bokeh-demo.png" width="50%">

Recurrent neural networks

Generating text

