from NeuralNetwork import Neural_Network
from sklearn.datasets import load_breast_cancer
import random
from GD_class import *
from sklearn.linear_model import LogisticRegression

"""
This script runs the testing of non preprocessed data to investigate why they give better results.
"""

def accuracy_test(y_true, y_pred):
    # y_true is the true labels of the data
    # y_pred is the predicted labels of the data by the neural network
    # Both y_true and y_pred are numpy arrays of the same shape

    # Compare y_true and y_pred element-wise and count the number of matches
    matches = 0

    for true, pred in zip(y_true, y_pred):
        if true == pred:
            matches = matches + 1

    # Calculate the accuracy as the ratio of matches to the total number of data points
    accuracy = matches / len(y_true)

    # Return the accuracy as a percentage
    return accuracy * 100

random.seed(367)

# Load the data
cancer = load_breast_cancer()
X_orig = cancer.data
target_true = cancer.target
target_true = target_true.reshape(target_true.shape[0], 1)

# design NN
n_hidden_layers = 1
n_hidden_nodes = 2
n_outputs = 1

#training NN with cross entropy and softmax
NN_output_cross_entropy = []

for X_row, target_row in zip(X_orig, target_true):

    X = jnp.array([X_row])
    target = jnp.array([target_row])

    nn = Neural_Network(X, target, n_hidden_layers, n_hidden_nodes, n_outputs,
                        learning_rate=0.00001, lmbd=0.1, cost_function='CostCrossEntropy',
                        activation_function='softmax',
                        classification_problem=True)

    nn.feed_forward()
    nn.feed_backward()
    nn.train(num_iter=100)

    NN_output_cross_entropy.append(nn.output_layer.output)

    nn.reset_weights()


accuracy_cross_entropy= accuracy_test(target_true, NN_output_cross_entropy)
print(f'Accuracy score of NN cross entropy no preprocessing: {accuracy_cross_entropy}')

# training NN with logistic regression and sigmoid
NN_output_logreg = []

for X_row, target_row in zip(X_orig, target_true):
    X = jnp.array([X_row])
    target = jnp.array([target_row])

    nn = Neural_Network(X, target, n_hidden_layers, n_hidden_nodes, n_outputs,
                        learning_rate=0.00001, lmbd=0.1, cost_function='LogReg',
                        activation_function='sigmoid',
                        classification_problem=True)

    nn.feed_forward()
    nn.feed_backward()
    nn.train(num_iter=100)

    NN_output_logreg.append(nn.output_layer.output)

    nn.reset_weights()

accuracy_logreg = accuracy_test(target_true, NN_output_logreg)
print(f'Accuracy score of NN logistic regression no preprocessing: {accuracy_logreg}')

# NN setup test with no layers for logistic regression
n_hidden_layers = 0
n_hidden_nodes = 0
n_outputs = 1

# testing no preprosseing for a NN with no hidden layers
NN_output_logreg_no_hidden_layers = []

for X_row, target_row in zip(X_orig, target_true):
    X = jnp.array([X_row])
    target = jnp.array([target_row])

    nn = Neural_Network(X, target, n_hidden_layers, n_hidden_nodes, n_outputs,
                        learning_rate=0.00001, lmbd=0.1, cost_function='LogReg',
                        activation_function='sigmoid',
                        classification_problem=True)

    nn.feed_forward()
    nn.feed_backward()
    nn.train(num_iter=100)
    NN_output_logreg_no_hidden_layers.append(nn.output_layer.output)

    nn.reset_weights()

accuracy_logreg_no_hidden = accuracy_test(target_true, NN_output_logreg_no_hidden_layers)
print(f'Accuracy score of NN no hidden layers and no preprocessing: {accuracy_logreg_no_hidden}')

# testing no preprocessing scikit
cancer = load_breast_cancer()
# Logistic Regression
logreg = LogisticRegression(solver='lbfgs')
logreg.fit(cancer.data,cancer.target)
print("Test set accuracy with scikit Logistic Regression no preprossesing: {:.2f}".format(logreg.score(cancer.data,cancer.target)))
