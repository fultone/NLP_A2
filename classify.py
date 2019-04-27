'''
Starter code for A2
'''
import argparse
import nltk
import numpy as np
import tqdm
import sklearn.metrics
import collections
from collections import defaultdict


global _TOKENIZER
_TOKENIZER = nltk.tokenize.casual.TweetTokenizer(
    preserve_case=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_docs',
                        type=str,
                        default='train.txt',
                        help='Path to training documents')
    parser.add_argument('--val_docs',
                        type=str,
                        default='val.txt',
                        help='Path to validation documents')
    return parser.parse_args()


def tokenize(string):
    '''Given a string, consisting of potentially many sentences, returns
    a lower-cased, tokenized version of that string.
    '''
    global _TOKENIZER
    return _TOKENIZER.tokenize(string)


def load_labeled_corpus(fname):
    '''Loads a labeled corpus of documents'''
    documents, labels = [], []
    with open(fname) as f:
        for line in tqdm.tqdm(f):
            if len(line.strip()) == 0: continue
            label, document = line.split('\t')
            labels.append(int(label))
            documents.append(tokenize(document))
    return documents, np.array(labels)


''' ************************** HOMEWORK #1 ************************** '''
def classify_doc_hand_design(doc_in, valid_words=[('good', 1), ('bad', -1),
                                ('excellent', 1), ('dissapointing', -1)]):
    score = 0
    for list in doc_in:
        for i in list:
            for j in valid_words:
                if j[0] == i:
                    score += j[1]

    if score > 0:
        return 1
    else:
        return 0

''' ************************** HOMEWORK #2 ************************** '''
# Returns a dictionary of all unique words in docs and their coorisponding count
def getVocab(docs):
    vocab = defaultdict(int)
    for doc in docs:
        for word in doc:
            if vocab[word] == 0:
                vocab[word] = 1
            else:
                vocab[word] += 1
    return vocab


def fillCounts(docs, labels, sentiment):
    return 0

def get_nb_probs(train_docs, train_labels, smooth=1.0):
    n_pos = 0   # the number of positive documents
    n_neg = 0   # the number of negative documents
    for label in train_labels:
        if label == 0:
            n_neg += 1
        else:
            n_pos += 1

    vocab = getVocab(train_docs)


    pos_counts = defaultdict(int) # a dictionary mapping {word w: P(w | pos_class)}
    neg_counts = defaultdict(int) # a dictionary mapping {word w: P(w | neg_class)}

    # fill pos_probs and neg_probs
    for i in range(0,len(train_docs)):
        for j in range(0, len(train_docs[i])):
            if train_labels[i] == 1: # if the document is positive
                if pos_counts[train_docs[i][j]] == 0:
                    pos_counts[train_docs[i][j]] = 1
                else:
                    pos_counts[train_docs[i][j]] += 1
            elif train_labels[i] == 0:
                if neg_counts[train_docs[i][j]] == 0:
                    neg_counts[train_docs[i][j]] = 1
                else:
                    neg_counts[train_docs[i][j]] += 1

    pos_probs = defaultdict(int) # a dictionary mapping {word w: P(w | pos_class)}
    neg_probs = defaultdict(int) # a dictionary mapping {word w: P(w | neg_class)}

    for word in vocab:
        pos_probs[word] = (pos_counts[word] + smooth) / (n_pos + len(vocab)*smooth)
        neg_probs[word] = (neg_counts[word] + smooth) / (n_neg + len(vocab)*smooth)

    return pos_probs, neg_probs, n_pos, n_neg

def classify_doc_naive_bayes(doc_in, pos_probs, neg_probs, n_pos, n_neg):
    '''Given an input document and the outputs of get_nb_probs, this
    function computes the summed log probability of each class given
    the input document, according to naive bayes. If the token-summed
    positive log probability is greater than the token-summed negative
    log probability, then this function outputs 1. Else, it outputs 0.
    '''
    log_prob_pos = 0
    log_prob_neg = 0

    for word in doc_in:
        if(pos_probs[word] != 0):
            log_prob_pos += np.log(pos_probs[word])
        if(neg_probs[word] != 0):
            log_prob_neg += np.log(neg_probs[word])

    log_prob_pos += np.log(n_pos/(n_pos+n_neg))
    log_prob_neg += np.log(n_neg/(n_pos+n_neg))
    if log_prob_pos < log_prob_neg:
        return 0
    return 1


''' ************************** HOMEWORK #3 ************************** '''
def get_logistic_regression(train_docs, train_labels, min_vocab_occur=20):

    import sklearn.linear_model
    import sklearn.preprocessing
    import sklearn.pipeline

    model = sklearn.pipeline.Pipeline([('scaler', sklearn.preprocessing.StandardScaler()),
                                       ('model', sklearn.linear_model.LogisticRegressionCV(Cs=5))])

    word_counts = collections.Counter()
    for t in train_docs:
        word_counts.update(t)

    vocab = [w for w, c in word_counts.items() if c >= min_vocab_occur]
    word2idx = {w:idx for idx, w in enumerate(vocab)}

    data_matrix = np.zeros((len(train_docs), len(vocab)))

    countAnd = 0
    for doc_idx, doc in enumerate(train_docs):
        if(doc_idx == 33):
            for word in doc:
                if word == "and":
                    countAnd += 1
        for t in doc:
            if t in word2idx:
                data_matrix[doc_idx, word2idx[t]] += 1

    ## Q1: How many times does the word "and" appear in document 33?
    print("Question 1. The word 'and' appears in document #33 a total of", countAnd, "times.")

    model.fit(data_matrix, train_labels)
    ## Q2: Describe in words what this model does to classify
    ## documents. Your answer should use the words "scale", "weights",
    ## "bias", "sigmoid", and "pipeline".
    '''
        This function returns "word2idx," a dictionary mapping all words in
    the vocabulary to indicies, as well as "model," a call to the Pipeline
    function which passes in both a call to the standard scalar class and
    logistic regression class on the given data.
        Note:
        * The "StandardScaler" class standardizes (ie normally distributes)
    the data by removing the mean value and scaling to unit variance. The class
    uses a biased estimator to calculate the standard deviation.
        * The "logisticRegression" class implements logistic regression using
    built-in optimizers. It selects hyperparameters via cross-validation. In
    this example, the parameter "fit_intercept" is set to True, meaning a bias
    is added to the decision function.
        * The "Pipeline" class allows the user to assemble several steps
    (ie standardscalar, logisticregression) for a given data set, that can be
    cross-validated together, while setting different parameters.
    '''
    return word2idx, model

''' ************************** HOMEWORK #4 ************************** '''
def classify_doc_logistic_regression(doc_in, vocab_lr, model_lr):
    '''This function builds a term count vector (numpy array) from doc_in,
    according to vocab_lr (logistic regression), which is a dictionary mapping
    words to indices in the vocabulary. model_lr is the sklearn pipeline built
    in the get_logistic_regression function. Note that you can call
    model_lr.predict(x) to generate model predictions for vectors x.
    '''
    # make the feature vector, then call model.predict on that feature vector
    featureVector = np.zeros((1,len(vocab_lr)))
    for word in doc_in:
        index = vocab_lr[word]
        featureVector[index] += 1

    return model_lr.predict(featureVector)

''' ************************** HOMEWORK #5 ************************** '''
def get_accuracy(true, predicted):
    arr = np.equal(true, predicted)
    num_correct = np.count_nonzero(np.where(arr==True, 1, arr))
    accuracy = (num_correct) / len(predicted)
    return accuracy

# true_pos = where (predicted = 1) and (true = 1)
# false_pos = where (predicted = 1) and (true = 0)
def get_precision(true, predicted):
    together = np.array([true, predicted])
    true_pos = (np.sum(together, axis=0) == 2).sum())
    false_pos = (np.subtract(true, predicted) == -1).sum()
    return (true_pos) / (true_pos + false_pos)

# true_pos = where (predicted = 1) and (true = 1)
# false_neg = where (predicted = 0) and (true = 1)
def get_recall(true, predicted):
    together = np.array([true, predicted])
    true_pos = (np.sum(together, axis=0) == 2).sum())
    false_neg = (np.subtract(predicted, true) == -1).sum()
    recall = (true_pos) / (true_pos + false_neg)
    return recall

def get_f1(true, predicted):
    precision = get_precision(true, predicted)
    recall = get_recall(true, predicted)
    return (2 * precision * recall) / (precision + recall)


def classify_doc_constant(doc_in):
    '''Constant prediction classifier'''
    return 0

def get_metrics(true, predicted):
    accuracy = get_accuracy(true, predicted)
    precision = get_precision(true, predicted)
    recall = get_recall(true, predicted)
    f1 = get_f1(true, predicted)
    return accuracy, precision, recall, f1

''' ****************************** MAIN ****************************** '''
def main():
    args = parse_args()
    train_docs, train_labels = load_labeled_corpus(args.train_docs)
    val_docs, val_labels = load_labeled_corpus(args.val_docs)


    print('Label statistics, n_pos/n_total: {}/{}'.format(
        np.sum(train_labels==1), len(train_labels)))
    print('Label statistics, n_pos/n_total: {}/{}'.format(
        np.sum(val_labels==1), len(val_labels)))

    ## Naive bayes
    pos_probs, neg_probs, n_pos, n_neg = get_nb_probs(train_docs, train_labels)
    nb_predictions = np.array([classify_doc_naive_bayes(d, pos_probs, neg_probs, n_pos, n_neg)
                               for d in val_docs])


    vocab_lr, model_lr =  get_logistic_regression(train_docs, train_labels)
    print(model_lr)

    ## Constant prediction
    constant_predictions = np.array([classify_doc_constant(v) for v in val_docs])

    ## Hand-designed classifier prediction
    hand_predictions = np.array([classify_doc_hand_design(v, valid_words=[('good', 1), ('bad', -1), ('excellent', 1),('dissapointing', -1)])

                                 for v in val_docs])



    vocab_lr, model_lr =  get_logistic_regression(train_docs, train_labels)
    lr_predictions = np.array([classify_doc_logistic_regression(d, vocab_lr, model_lr)
                               for d in val_docs])

    # NLP folks sometimes multiply metrics by 100 simply for aesthetic reasons
    print(' & '.join(['{:.2f}'.format(100*f)
                      for f in get_metrics(val_labels, constant_predictions)]))
    print(' & '.join(['{:.2f}'.format(100*f)
                      for f in get_metrics(val_labels, hand_predictions)]))
    print(' & '.join(['{:.2f}'.format(100*f)
                      for f in get_metrics(val_labels, nb_predictions)]))
    print(' & '.join(['{:.2f}'.format(100*f)
                      for f in get_metrics(val_labels, lr_predictions)]))


if __name__ == '__main__':
    main()
