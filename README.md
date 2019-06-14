# Project for dl4nlp: Irony detection and domain adaptation #

irony_detection.py

The subfolder irony_data contains two corpora annotated for irony:
* a larger out-of-domain set containing tweets (as used for this SemEval task: https://competitions.codalab.org/competitions/17468)
* a smaller in-domain set containing reddit comments (as provided on https://www.kaggle.com/rtatman/ironic-corpus)

The script trains 5 different models on this data trying to optimize the performance on the smaller dataset:
* baseline model 1: trained solely on out-of-domain data (larger corpus)
* baseline model 2: trained solely on in-domain data (smaller corpus)
* joint model:      trained on a concatenation of in-domain and out-of-domain data
* weighted model:   trained on a weighted concatenation of in-domain and out-of-domain data
* continued model:  baseline model 1 trained further on in-domain data

All models are saved to a subfolder (models) and the results of their evaluation in terms of
accuracy, precision, recall and F1-score is written to scores.txt in the project folder.

The script needs to be called from within the project folder and requires
python 3.5
pandas 0.22.0
numpy 1.13.3
nltk 3.2.5
scikit-learn 0.19.1.
tensorflow 1.4.1
keras 2.1.3
h5py 2.7.1
(previous versions might work as well)