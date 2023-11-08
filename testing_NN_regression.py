from Gridsearch import GridSearch_LinReg_epochs_batchsize
from dataframes import df_analysis_method_is_index
from plotting import plot_SGD_MSE_convergence_epoch_batch
from GD_class import *
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler
import jax.numpy as jnp
from sklearn.preprocessing import StandardScaler
from NeuralNetwork import Neural_Network
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import  train_test_split
from sklearn.datasets import load_breast_cancer
import pandas as pd
import random
import seaborn as sns


"""
Simple script to test that the NN works for linear regression problems ( not included in paper)
The neural networks give out good results for MSE ( around 0.8) but bad results for R2 (around -4)

"""

"""
Functions needed
"""

def cost_function_OLS(X, y, beta):
    n = len(y)  # Define the number of data points
    return (1.0/n) * jnp.sum((y - jnp.dot(X, beta))**2)

def analytical_gradient(X, y, beta):
    n = len(y)
    return (2.0/n)*jnp.dot(X.T, ((jnp.dot(X, beta))-y))

def make_design_matrix(x, degree):
    "Creates the design matrix for the given polynomial degree and ijnput data"

    X = np.zeros((len(x), degree + 1))

    for i in range(X.shape[1]):
        X[:, i] = np.power(x, i)

    return jnp.array(X)

"""
generate data and design matrix
"""

np.random.seed(1342)


n = 1000

x = jnp.linspace(0, 1, n)
y = jnp.sum(jnp.asarray([x ** 2]), axis=0) + 0.2 * np.random.normal(size=len(x))


# Making a design matrix to use for linear regression part
degree = 2
X = make_design_matrix(x, degree)

#splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


#now scale the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

yscaler = StandardScaler()
yscaler.fit(y_train.reshape(-1, 1))
y_train_scaled = yscaler.transform(y_train.reshape(-1,1))


# to test outcome
true_beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train


# Set parameters
learning_rate = 0.1
tol=1e-3
momentum=0.5
delta= 1e-8
rho1 = 0.9
rho2 = 0.99

# NN setup
n_hidden_layers = 1
n_hidden_nodes = 2
n_outputs = 1
learning_rate=0.1


def MSE(true_y, predicted_y):
    '''
    Calculates the the mean squared error (MSE) of the model.
    In essence we calculate the average error for the predicted y_i's compared to the true y_i's.

    :param true_y:
    :param predicted_y:
    :return:
    '''

    n = len(true_y)
    # predicted_y = predicted_y.reshape(true_y.shape)
    SSR = np.sum((true_y - predicted_y) ** 2)  # Residual som of squares, measures the unexplained variability ("errors")
    MSE = (1 / n) * SSR
    return MSE

def R_squared(true_y, predicted_y):
    '''
    Calculates the coefficient of determination R^2. R^2 quantifies the proportion of total variability in the
    dataset that is explained by the model.

    :param true_y:
    :param predicted_y:
    :return:
    '''
    # predicted_y = predicted_y.reshape(true_y.shape)
    mean_true_y = jnp.mean(true_y)
    SSR = jnp.sum((true_y - predicted_y) ** 2)  # Residual som of squares, measures the unexplained variability ("errors")
    TSS = jnp.sum((true_y - mean_true_y) ** 2)  # Total sum of squares, measures the total variability

    R2 = 1 - SSR / TSS

    return R2


#training NN with breastcancer data
NN_output_train = []
NN_output_test = []

for X_row_train, X_row_test, target_row in zip(X_train_scaled, X_test_scaled, y_train_scaled):

    X = jnp.array([X_row_train])
    target = jnp.array([target_row])

    nn = Neural_Network(X, target, n_hidden_layers, n_hidden_nodes, n_outputs,
                        learning_rate=learning_rate, cost_function='CostOLS',
                        activation_function='sigmoid',
                        classification_problem=False, activate_output=False)

    nn.feed_forward()

    nn.feed_backward()
    nn.train(num_iter=50)

    X_pred = jnp.array([X_row_test])
    test_pred = nn.predict(X_pred)

    NN_output_train.append(nn.output_layer.output)
    NN_output_test.append(test_pred)

    nn.reset_weights()

test_pred =  jnp.array([NN_output_test]).ravel()


## skalere y dataen også?
MSE_row = MSE(y_test, test_pred)
R2_Row = R_squared(y_test, test_pred)
true_pred = X_test_scaled@true_beta
MSE_true = MSE(y_test, true_pred)
R2_true = R_squared(y_test, true_pred)

print(MSE_row)
print(R2_Row)
print(MSE_true)
print(R2_true)