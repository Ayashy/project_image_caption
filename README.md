# Preface

In this project, we try to implement an image captioning systeme base on the research paper [Show, Attend, and Tell](https://arxiv.org/abs/1502.03044) using Python 3.7 and Pytorch.

# Project Structure

### General project structures

```
Documentation   : Contains Docs/Source code usefull for project developpement
Src             : Contains the main project files
|_ Soft         : Source files for the soft approach
|_ Hard         : Source files for the hard approach
Tutorials       : Exemple projects used to practice the concepts used in the project
```

### Data folders

The data must be stored according to this structure in order for the data processing functions to run

```
Soft            
|_ raw_data             : Contains the flikr datafiles
    |_ Flickr8k_data    : Put the jpg images here
    |_ Flickr8k_text    : Put the caption text files here
|_ processed_data       : Generated input files will be stored in this folder
    |_ eval_images      : Validation image features
    |_ test_images      : Test image features
    |_ train_images     : Train image features
```


# Usefull links

### Theory

- http://kelvinxu.github.io/projects/capgen.html : 
Main article paper
- https://www.quora.com/What-is-the-VGG-neural-network:
Infos about VGG model
- https://colah.github.io/posts/2015-08-Understanding-LSTMs/
Best ressource to understand Neural networks imo.
- https://www.youtube.com/watch?v=WCUNPb-5EYI
Video explaining how lstm works 
- https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html
Article explaining LSTM initialisation
- https://jhui.github.io/2017/03/15/Soft-and-hard-attention/
Article explaining attention models
- https://blog.heuritech.com/2016/01/20/attention-mechanism/
Another article about attention models

### Implementation tutorials


- http://www.jessicayung.com/lstms-for-time-series-in-pytorch/ :
How to implement an lstm in pytorch
- https://www.analyticsvidhya.com/blog/2018/04/solving-an-image-captioning-task-using-deep-learning/
Another LSTM tutorial
- https://machinelearningmastery.com/prepare-photo-caption-dataset-training-deep-learning-model/
Good tutorial about loading data for caption generation
- https://hackernoon.com/understanding-architecture-of-lstm-cell-from-scratch-with-code-8da40f0b71f4
How to implement an LSTM from scratch
- https://github.com/tangbinh/image-captioning
Good repository that implements the main article's model using pytorch
- https://github.com/coldmanck/show-attend-and-tell
Another repo that uses tensorflow
- https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/models.py
Good repo with great explanation on the readme

