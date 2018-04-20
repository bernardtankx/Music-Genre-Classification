# Music Genre Classification using Multi-Channel Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM)
These codes are written as part of a project for Music Genre Classification for CS4347 Sound and Music Computing module. The dataset used is provided by the module. 

This repository contains 2 source directories:
* Multi-Channel Convolutional Neural Network (MCC)
* Multi-Channel Convolutional Neural Network and Long Short-Term Memory (MCCLSTM)

Both directories contained their respective preprocessing script, `preprocessing.py` and training script, `train.py`.

## Disclaimer
Note that the codes written in this repository are based on [1] and [2], however they are not meant to be representative of the research papers.

## Usage
The code can be run using the following command to generate the numpy files for the dataset.

`python preprocessing.py`

For training of MCC and MCCLSTM, the following command are used. 

Format: [SCRIPT] [TRAIN] [CHANNEL]

`python train.py 2`

`python train.py 3`

## Evaluation
Based on the given training data, 10-fold cross validation with a split of 80%-10%-10%, for the train-validation-test set, will be used to evaluate the performance of the networks. For MCC, since each song is splitted into 16 chunks, the majority voting approach is used to determine the genre of each particular song. For MCCLSTM, the entire song are input into the network. Therefore, majority voting approach is not used.

| Convolutional Neural Network  | Accuracy: mean ± std (%)  |
|-------------------------------|:-------------------------:|
| 2-Channel MCC                 |       78.60 ± 2.11        |
| 3-Channel MCC                 |       80.30 ± 1.91        |
| 2-Channel MCCLSTM             |       60.00 ± 1.40        |
| 3-Channel MCCLSTM             |       62.10 ± 2.00        |

## References
[1] Pons, Jordi, Thomas Lidy, and Xavier Serra. "Experimenting with musically motivated     convolutional neural networks." Content-Based Multimedia Indexing (CBMI), 2016 14th International Workshop on (2016): 1-6

[2] Ning Chen and Shijun Wang. “High-Level Music Descriptor Extraction Algorithm based on Combination of Multi-Channel CNNs and LSTM” Proceedings of the 18th International Society for Music Information Retrieval Conference (2017): 509-514
