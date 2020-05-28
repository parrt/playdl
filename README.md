# Exploring deep learning neural networks

This repository is just a collection of notebooks that I'm using to explore various deep learning (DL) concepts and APIs (such as keras).  The mathematics behind the networks is straightforward but tedious so I'm focusing on applying networks.

## AWS stuff

* [Remote jupyter lab access](aws-setup.md)

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
<img src="images/bokeh-demo.png" width="35%">

I'm going to start again on collaborative filtering with pytorch: [Regularized collaborative filtering with pytorch](notebooks/collaborative-filtering-regularized.ipynb). **Lessons:**

* I was only tracking the training error, which I could get pretty low but then I was wondering why the validation error using a random forest OOB (user+movie->rating) often was not good. Duh.  I'm watching the training error inside the training loop, not the validation error.
* By randomly selecting validation and training sets at each iteration (epoch), we are moving to stochastic gradient descent (SGD). Previously I only used gradient descent.
* I could not get the bias version working well. Yannet pointed out that I need to initialize my randomized weights so that the predicted rating has an average of 2.5 or somewhere else between zero and five. By the way, adding bias dramatically slowed the training process down! Way more than I would expect, given the small increase in data and operations. 
* I also noticed that Jeremy/Sylvain does L2 regularization in their book, which clued me in to the fact that validation error is important. We don't want those coefficients getting too big as they are likely chasing something in the training set.
* I also found that simple gradient descent with momentum did a pretty good job and I didn't need the complexity of AdaGrad.
* Another trick from fastai book is to keep the predicted rating in the range 0..5, which constrains where the loss function can send gradient descent.

## Recurrent neural networks

## Generating text

