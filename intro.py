'''CS322 Introduction to numpy, vectorization, and machine learning in python

please fill in your answers either as print statements (Q1, Q2, Q3,
Q5, Q6) or function implementations (Q4). Your answers should be no
more than a few lines long, and contain *no python for loops!* For
what it's worth --- it's possible to answer each question in a single
line.

'''

import numpy as np

### Part 1: The data

data_matrix = np.load('data_matrix.npy')
labels = np.load('labels.npy')

# Explore the shapes of the data

## Q1: how many data points are there in this dataset?
print("1. NUMBER OF DATA POINTS:", data_matrix.shape[0])

## Q2: how many features does each datapoint have?
print("\n2. NUMBER OF FEATURES PER DATA POINT:", data_matrix.shape[1])

## Q3: what is the mean value, over all of the data points, of
## the feature at index 2? The code that generates this answer
## should contain no explicit python loops, i.e, it should
## rely on numpy vectorization
print("\n3. AVERAGE OF FEATURE AT INDEX 2:", np.sum(data_matrix[2])/data_matrix.shape[1])

### Part 2: the sigmoid function
my_array = np.array([-1.0 ,0.0, 1.0])

## Q4: write a function that computes the sigmoid of each element in a numpy array
## your function should contain no explicit python loops, i.e., it should rely on
## numpy vectorization
def sigmoid(x):
    # 1 / (1 + e ^ -x)
    #x = np.exp(-x) | x = np.add(x,1) | x = np.divide(1, x)
    return np.divide(1, np.add(np.exp(-x),1))

print("\n4. COMPUTE THE SIGMOID OF X:", sigmoid(my_array))
# the answer I get: [0.26894142, 0.5, 0.73105858]

### Part 3: Logistic regression model

# assume that these are the weights of a logistic regression
# model learned according to gradient descent.
W, b = np.load('logistic_regression_weights.npy'), np.load('logistic_regression_bias.npy')
## Q5: What are the probabilities of label=1 for each datapoint,
## according to the logistic regression model?
# Likelihood(w,b | x,y) = P(y = 1|W,b,x_i)
#                       = o(w DOT x_i + b)^1
numpy_probs = sigmoid(np.add(data_matrix.dot(W), b))
print("\n5. PROBABILITIES (WHEN LABEL=1):", numpy_probs)

## Q6: Assume that any estimated probability over .5 is a guess of
## label=1, and any estimated probability under .5 is a guess of
## 0. What is the accuracy of the model with respect to the labels?

#accuracy = [1 for x in numpy_probs if x > 0.5]
#accuracy = [0 for x in numpy_probs if x < 0.5]
print("\n6. ACCURACY OF THE MODEL:",)
rounded = np.round(numpy_probs)
#if 1 and true or if 0 and false ==> correct, else not correct

print(labels)
# how many times the thing I answered = labels

## Here is how logistic regression is implemented in Keras:
import keras

logistic_layer = keras.layers.Dense(
    1, # we only want 1 number output, i.e., the probability.
    input_shape=(data_matrix.shape[1],), # how many input features there are
    weights=[np.expand_dims(W,1), b], # this gives the layer the weights we loaded
    activation='sigmoid')
logistic_regression_model = keras.models.Sequential([logistic_layer])
keras_probs = logistic_regression_model.predict(data_matrix)

print('These should match:')
print(keras_probs.flatten())
print(numpy_probs.flatten())
