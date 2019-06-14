#!/usr/bin/env python

"""
irony_detection.py
This script trains and evaluates 5 different models for irony detection based on two distinct datasets
of different sizes (provided in the irony_data subfolder).

It saves the trained models to a subdirectory (models) and outputs an evaluation file (scores.txt) containing
accuracy, precision, recall and F1-score on a test set drawn from the smaller dataset for each of the models
in order to explore if the larger dataset can be used to boost the performance on the smaller dataset.

Author: Anne Beyer (11802814)
Date: 28.01.2018
"""

import os
import numpy as np
from collections import Counter
import pandas as pd  # data processing for CSV files
import random
import math
from nltk.tokenize import TweetTokenizer
from sklearn.utils import class_weight  # automatically determine class_weights for unbalanced data sets
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Bidirectional
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

# seeds for reproducibility
random.seed(23)
np.random.seed(42)

# parameter definitions
UNK_TOKEN = "<UNK>"
EMBEDDING_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 100  # will be less with early stopping
HIDDEN_SIZE = 10
MAX_LENGTH = 100  # the average word_count in the kaggle corpus is 41.2, the maximum is 800
#VOCAB_SIZE will be determined by create_dictionary

tokenizer = TweetTokenizer(preserve_case=False)  # lower-cases at the same time

# create model directory
model_path = "models"
if not os.path.exists(model_path):
    os.makedirs(model_path)


# function definitions

def read_kaggle_data():
    '''
    Read in in-domain data from csv file (comment,label)
    :return:  a list with the tokenized, lowercased comments (kaggle_corpus) and
    the corresponding irony labels (kaggle_lables, 0 or 1)
    '''
    # ! data is imbalanced (-1: 1412, 1: 537)

    kaggle_data = pd.read_csv('irony_data/kaggle-irony-labeled.csv', encoding='utf-8')
    corpus = kaggle_data.columns[0]
    labels = kaggle_data.columns[1]

    # adjust labels to SemEval dataset (because 0/1 is easier to model with sigmoid than -1/1)
    kaggle_labels = np.asarray(kaggle_data[labels])
    kaggle_labels[np.where(kaggle_labels == -1)] = 0
    # tokenize corpus (with same tokenizer as SemEval data for consistency)
    kaggle_corpus = [tokenizer.tokenize(comment) for comment in kaggle_data[corpus]]

    return kaggle_corpus, kaggle_labels


def read_semEval_data():
    '''
    Read in out-of-domain data from txt file (id    label   tweet)
    :return: a list with the tokenized, lowercased tweets (semEval_corpus) and
    the corresponding irony labels (semEval_lables, 0 or 1)
    '''
    semEval_corpus = []
    semEval_labels = []

    with open('irony_data/SemEval2018-T3-train-taskA.txt', 'rt') as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"):  # discard first line if it contains metadata
                line = line.rstrip()  # remove trailing whitespace
                label = int(line.split("\t")[1])
                tweet = line.split("\t")[2]
                semEval_labels.append(label)
                # tokenize with tweet tokenizer from nltk
                semEval_corpus.append(tokenizer.tokenize(tweet))

    return semEval_corpus, semEval_labels


def create_dictionary(corpus):
    '''
    Creates a dictionary that maps words to ids.
    The dictionary contains only words with frequency > 1 and a placeholder '<unk>' for unknown words.
    The place holder has the id 0.
    :param corpus: corpus from which the dictionary is created
    :return: dictionary that maps words to ids
    '''
    # construct vocabulary from corpus
    vocab_freq = Counter()
    for comment in corpus:
        vocab_freq.update(comment)

    # only keep words with freq > 1
    vocab = [k for k in vocab_freq if vocab_freq[k] > 1]
    # set global VOCAB_SIZE
    global VOCAB_SIZE
    VOCAB_SIZE = len(vocab) + 1

    # add ids and unk token with id 0
    word_to_id = {w: (i + 1) for i, w in enumerate(vocab)}
    word_to_id[UNK_TOKEN] = 0

    return word_to_id


def prepare_corpus(dictionary, corpus):
    '''
    Maps words to indices and pads all sequences to equal lengths
    :param dictionary: word to id map
    :param corpus: corpus to be processed
    :return: processed corpus
    '''
    # map words to indices from dictionary
    corpus = [to_ids(segment, dictionary) for segment in corpus]
    # pad sequences to equal length
    prepared_corpus = pad_sequences(corpus, maxlen=MAX_LENGTH, padding='post', truncating='post', value=0)
    return prepared_corpus


def to_ids(words, dictionary):
    '''
    Takes a list of words and converts them to ids using the word2id dictionary.
    :param words: the list to be converted
    :param dictionary: the word to id mapping
    :return: the converted list
    '''
    return [dictionary.get(word, dictionary[UNK_TOKEN]) for word in words]


# create train dev and test sets
def random_train_dev_test_split(num_total_items, train_ratio=0.5, dev_ratio=0.25):
    '''
    Calculate random splits into train dev and test set
    :param num_total_items: the total number of available segments
    :param train_ratio: the ratio for the training set
    :param dev_ratio: the ratio for the dev set
    :return: three arrays containing the ids of the respective split in the overall set
    '''
    num_train_items = math.floor(num_total_items * train_ratio)
    num_dev_items = math.floor(num_total_items * dev_ratio)
    num_test_items = num_total_items - num_train_items - num_dev_items
    split = [0] * num_train_items + [1] * num_dev_items + [2] * num_test_items
    random.shuffle(split)
    split = np.asarray(split)
    return split == 0, split == 1, split == 2


def unison_shuffled(a, b, c):
    '''
    Shuffle three lists the same way
    :param a: list 1
    :param b: list 2
    :param c: list 1
    :return: a unison permutation of the three lists
    '''
    assert len(a) == len(b)
    assert len(a) == len(c)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]


def calculate_class_weights(labels):
    '''
    Determine the class weights for an imbalanced set of labels
    :param labels: the imbalanced set of labels
    :return: a dict with the weights for these labels
    '''
    class_weights = class_weight.compute_class_weight('balanced', np.unique(labels), labels)
    return dict(enumerate(class_weights))


def train_and_evaluate_model(x_train, y_train, model_prefix, sample_weights=None, class_weights=None, old_model=None):
    '''
    Create, train and evaluate a model with the given parameters
    :param x_train: the training data
    :param y_train: the training labels
    :param model_prefix: the model name
    :param sample_weights: the weights for the training samples
    :param class_weights: the weights for the training classes
    :param old_model: the old model to be trained further
    :return: the model file
    '''
    # create specific filename for model
    model_file = "{}/{}_model.h5".format(model_path, model_prefix)

    irony_model = Sequential()

    if old_model is not None:
        # load old model for continuing training
        irony_model = load_model(old_model)
    else:
        # create new model
        irony_model.add(Embedding(VOCAB_SIZE, EMBEDDING_SIZE))
        irony_model.add(Bidirectional(LSTM(HIDDEN_SIZE)))
        irony_model.add(Dense(1, activation="sigmoid"))
        irony_model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

    # monitor loss on dev set (val_loss) for early stopping
    early_stopper = EarlyStopping(monitor='val_loss', patience=1, mode="min")
    checkpoint = ModelCheckpoint(model_file, monitor='val_loss', verbose=0, save_best_only=True,
                                 save_weights_only=False, mode='auto', period=1)

    if sample_weights is not None and class_weights is not None:
        irony_model.fit(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS,
                        validation_data=(x_dev, y_dev), sample_weight=sample_weights,
                        class_weight=class_weights, callbacks=[early_stopper, checkpoint])
    elif class_weights is not None:
        irony_model.fit(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS,
                        validation_data=(x_dev, y_dev), class_weight=class_weights,
                        callbacks=[early_stopper, checkpoint])

    elif sample_weights is not None:
        irony_model.fit(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS,
                        validation_data=(x_dev, y_dev), sample_weight=sample_weights,
                        callbacks=[early_stopper, checkpoint])
    else:
        irony_model.fit(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS,
                        validation_data=(x_dev, y_dev), callbacks=[early_stopper, checkpoint])

    evaluate_model(model_prefix, model_file)
    return model_file


def evaluate_model(model_prefix, model_file):
    '''
    Evaluate a given model and write accuracy, precision, recall and f1 score to the score file
    :param model_prefix: the name of the model to use
    :param model_file: the model to use
    '''
    model = load_model(model_file)
    predictions = model.predict(x_test)
    y_hat = predictions[:, 0]
    y_hat[np.where(y_hat < 0.5)] = 0
    y_hat[np.where(y_hat >= 0.5)] = 1
    acc = calc_accuracy(y_test, y_hat)
    p, r, f = precision_recall_fscore(y_test, y_hat)

    score_file.write("\nEvaluation of {} model\n".format(model_prefix))
    score_file.write("Accuracy:{0}\nPrecision:{1}\nRecall:{2}\nF1-score:{3}\n".format(acc, p, r, f))


def calc_accuracy(true, predicted):
    '''
    Calculates the accuracy of a classifier, defined as the fraction of correct classifications (code adapted from SemEval evaluate.py).
    :param true: the true labels
    :param predicted: the predicted labels
    :return:
    '''
    return sum([t == p for t, p in zip(true, predicted)]) / float(len(true))


def precision_recall_fscore(true, predicted):
    """Calculates the precision, recall and F-score of a classifier (code adapted from SemEval evaluate.py).
    :param true: iterable of the true class labels
    :param predicted: iterable of the predicted labels
    """
    labels = [0, 1]
    pos_label = 1
    # Build contingency table as ldict
    ldict = {}
    for l in labels:
        ldict[l] = {"tp": 0., "fp": 0., "fn": 0.}

    for t, p in zip(true, predicted):
        if t == p:
            ldict[t]["tp"] += 1
        else:
            ldict[t]["fn"] += 1
            ldict[p]["fp"] += 1

    # Calculate precision, recall and F-1 score per class
    for l, d in ldict.items():
        try:
            ldict[l]["precision"] = d["tp"] / (d["tp"] + d["fp"])
        except ZeroDivisionError:
            ldict[l]["precision"] = 0.0
        try:
            ldict[l]["recall"] = d["tp"] / (d["tp"] + d["fn"])
        except ZeroDivisionError:
            ldict[l]["recall"] = 0.0
        try:
            ldict[l]["fscore"] = 2 * ((ldict[l]["precision"] * ldict[l]["recall"]) / (
                    ldict[l]["precision"] + ldict[l]["recall"]))
        except ZeroDivisionError:
            ldict[l]["fscore"] = 0.0

    d = ldict[pos_label]
    return d["precision"], d["recall"], d["fscore"]


# main processing

# Open file for model evaluations
score_file = open('scores.txt', 'w')

# read in corpora
kaggle_corpus, kaggle_labels = read_kaggle_data()
semEval_corpus, semEval_labels = read_semEval_data()

# Create dictionary from in-domain data
word2id = create_dictionary(kaggle_corpus)

# map words to ids and pas sequences
kaggle_corpus_padded = prepare_corpus(word2id, kaggle_corpus)
semEval_corpus_padded = prepare_corpus(word2id, semEval_corpus)

# split in-domain data in train dev and test
train_spl, dev_spl, test_spl = random_train_dev_test_split(len(kaggle_corpus_padded))

# prepare in-domain data
x_train_kaggle = np.asarray(kaggle_corpus_padded[train_spl])
y_train_kaggle = kaggle_labels[train_spl]
kaggle_class_weights = calculate_class_weights(y_train_kaggle)
kaggle_sample_weights = np.full(len(x_train_kaggle), 4)

# prepare out-of-domain data
x_train_semEval = np.asarray(semEval_corpus_padded)
y_train_semEval = np.asarray(semEval_labels)
semEval_sample_weights = np.ones(len(x_train_semEval))

# prepare joint data
x_train_joint, y_train_joint, weights_joint = unison_shuffled(
    np.concatenate((x_train_kaggle, x_train_semEval)),
    np.concatenate((y_train_kaggle, y_train_semEval)),
    np.concatenate((kaggle_sample_weights, semEval_sample_weights)))

joint_class_weights = calculate_class_weights(y_train_joint)

x_train_joint2, y_train_joint2, weights_joint2 = unison_shuffled(
    np.concatenate((x_train_kaggle, x_train_kaggle, x_train_kaggle, x_train_kaggle, x_train_semEval)),
    np.concatenate((y_train_kaggle, y_train_kaggle, y_train_kaggle, y_train_kaggle, y_train_semEval)),
    np.concatenate((kaggle_sample_weights, kaggle_sample_weights, kaggle_sample_weights, kaggle_sample_weights, semEval_sample_weights)))

joint_class_weights2 = calculate_class_weights(y_train_joint2)

# prepare dev set (in-domain)
x_dev = kaggle_corpus_padded[dev_spl]
y_dev = kaggle_labels[dev_spl]

# prepare test set (in-domain)
x_test = kaggle_corpus_padded[test_spl]
y_test = kaggle_labels[test_spl]


# baseline model 1
# trained only on out-of-domain data (saving model for fine-tuning)
baseline_model = train_and_evaluate_model(x_train_semEval, y_train_semEval, "baseline_out_of_domain")

# baseline model 2
# trained only on in-domain data
train_and_evaluate_model(x_train_kaggle, y_train_kaggle, "baseline_in_domain", class_weights=kaggle_class_weights)

# joint model
# trained on combined data
train_and_evaluate_model(x_train_joint, y_train_joint, "joint", class_weights=joint_class_weights)

# weighted model
# trained on weighted combined data
#train_and_evaluate_model(x_train_joint, y_train_joint, "weighted", sample_weights=weights_joint)
# yields surprisingly bad reults, sample_weights seems to do somethng other than expected, manually 
# balancing data (below) shows expected improvements

# weighted model 2
# trained on a balanced combination of the data
train_and_evaluate_model(x_train_joint2, y_train_joint2, "joint2", class_weights=joint_class_weights2)

# continue model
# model trained on out-of-domain data first (= baseline model 1) and then
# further trained (fine tuned) on in-domain data
train_and_evaluate_model(x_train_kaggle, y_train_kaggle, "continue", class_weights=kaggle_class_weights,
                          old_model=baseline_model)


# close evaluation file
score_file.close()
print("\n\nFinished training models.\nResults can be found in scores.txt.\n")
