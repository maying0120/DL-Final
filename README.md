# Dense Video Caption Generation

This is the final project source code for the NYU course ECE-GY 9123 Deep Learning.

## Introduction

Our model is a transformer based EncoderDecoder model. Fed with multi-modality features, such as text features and video features etc, our model is able to encode the video information and then generate a short textual description of the given video clip.

## Train the model

```main.py``` contains the hyper parameters needed to train the model. ```dataset.py``` defines the format of the input data. ```loss.py``` has the loss function that we use and smoothing method that we implemented. In ```model.py```, we designed the architecture of our model as ```ContextModel```.

To train our model, you need to specify all the hyper parameters in the ```main.py```, then simply run

```
python3 main.py
```

## Evaluation

We use the open source work of [Microsoft COCO Caption Evaluation](https://github.com/ranjaykrishna/densevid_eval) to evaluate the result of the generated results.

## Contact information

For personal communication related to our model, please contact Xianglin Guo
(`kirk.guo@nyu.edu`), Ying Ma (`yingma@nyu.edu`).