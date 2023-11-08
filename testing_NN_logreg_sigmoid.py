from sklearn.preprocessing import StandardScaler
from NeuralNetwork import Neural_Network
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split
from sklearn.datasets import load_breast_cancer
import random
from GD_class import *
import seaborn as sns

"""
Simple script to test the NN with logistic regression as cost funtion and actiavtion 
function sigmoid.
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

#splitting
X_train, X_test, target_train, target_test = train_test_split(X_orig, target_true, test_size=0.20)
target_train = target_train.reshape(target_train.shape[0], 1)
target_test = target_test.reshape(target_test.shape[0], 1)

#now scale the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


"""
First testing for a NN with no hidden layers.
"""
# NN setup test with no layers for logistic regression
n_hidden_layers = 0
n_hidden_nodes = 0
n_outputs = 1

#training NN with breastcancer data
NN_output_cancer_train = []
NN_output_cancer_test = []

for X_row_train, X_row_test, target_row in zip(X_train_scaled, X_test_scaled, target_train):

    X = jnp.array([X_row_train])
    target = jnp.array([target_row])

    nn = Neural_Network(X, target, n_hidden_layers, n_hidden_nodes, n_outputs,
                        learning_rate=0.00001, lmbd=0.1, cost_function='LogReg',
                        activation_function='sigmoid', activate_output=False,
                        classification_problem=True)

    nn.feed_forward()
    nn.feed_backward()
    nn.train(num_iter=100)
    X_pred = jnp.array([X_row_test])
    test_pred = nn.predict(X_pred)

    NN_output_cancer_train.append(nn.output_layer.output)
    NN_output_cancer_test.append(test_pred)

    nn.reset_weights()


accuracy_breast_cancer_train_no_hidden_layers = accuracy_test(target_train, NN_output_cancer_train)
print(f'Accuracy score of training data NN logistic regression no hidden layers: {accuracy_breast_cancer_train_no_hidden_layers}')

accuracy_breast_cancer_test_no_hidden_layers = accuracy_test(target_test, NN_output_cancer_test)
print(f'Accuracy score of test data NN logistic regression no hidden layers: {accuracy_breast_cancer_test_no_hidden_layers}')


"""
Test with logistic regression with cost and activation functions
"""


n_hidden_layers = 1
n_hidden_nodes = 2
n_outputs = 1

#training NN with breastcancer data
NN_output_cancer_train = []
NN_output_cancer_test = []


for X_row_train, X_row_test, target_row in zip(X_train_scaled, X_test_scaled, target_train):

    X = jnp.array([X_row_train])
    target = jnp.array([target_row])

    nn = Neural_Network(X, target, n_hidden_layers, n_hidden_nodes, n_outputs,
                        learning_rate=0.0001,lmbd=0.1, cost_function='LogReg',
                        activation_function='sigmoid',
                        classification_problem=True)

    nn.feed_forward()

    nn.feed_backward()
    nn.train(num_iter=100)

    X_pred = jnp.array([X_row_test])
    test_pred = nn.predict(X_pred)

    NN_output_cancer_train.append(nn.output_layer.output)
    NN_output_cancer_test.append(test_pred)

    nn.reset_weights()


accuracy_breast_cancer_train_logreg = accuracy_test(target_train, NN_output_cancer_train)
print(f'Accuracy score of training data NN logistic regression cost function: {accuracy_breast_cancer_train_logreg}')

accuracy_breast_cancer_test_logreg = accuracy_test(target_test, NN_output_cancer_test)
print(f'Accuracy score of test data NN logistic regression cost function: {accuracy_breast_cancer_test_logreg}')

"""
Plot histograms
"""

fig, axes = plt.subplots(15,2,figsize=(10,20))

NN_output_cancer_test = jnp.array(NN_output_cancer_test).ravel()

malignant = X_test_scaled[NN_output_cancer_test == 0]

benign = X_test_scaled[NN_output_cancer_test == 1]

ax = axes.ravel()

for i in range(30):
    _, bins = np.histogram(X_test_scaled[:,i], bins = 50)
    ax[i].hist(malignant[:,i], bins = bins, alpha = 0.5)
    ax[i].hist(benign[:,i], bins = bins, alpha = 0.5)
    ax[i].set_title(cancer.feature_names[i])
    ax[i].set_yticks(())
ax[0].set_xlabel("Feature magnitude")
ax[0].set_ylabel("Frequency")
ax[0].legend(["Malignant", "Benign"], loc ="best")
fig.tight_layout()
plt.show()


"""
grid search for eta and lmb analysis
"""

eta_vals = np.logspace(-5, 1, 7)
lmbd_vals = np.logspace(-5, 1, 7)

test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))

for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):

        NN_output_cancer_train = []
        NN_output_cancer_test = []

        for X_row_train, X_row_test, target_row in zip(X_train_scaled, X_test_scaled, target_train):

            X = jnp.array([X_row_train])
            target = jnp.array([target_row])

            nn = Neural_Network(X, target, n_hidden_layers, n_hidden_nodes, n_outputs,
                                learning_rate=eta, lmbd=lmbd, cost_function='LogReg',
                                activation_function='sigmoid',
                                classification_problem=True)

            nn.feed_forward()
            nn.feed_backward()
            nn.train(num_iter=1)

            X_pred = jnp.array([X_row_test])
            test_pred = nn.predict(X_pred)

            NN_output_cancer_train.append(nn.output_layer.output)
            NN_output_cancer_test.append(test_pred)

            nn.reset_weights()

        print("Learning rate  = ", eta)
        print("Lambda = ", lmbd)
        print("Accuracy score on test set: ", accuracy_test(target_test, NN_output_cancer_test))
        print()
        test_accuracy[i][j] = accuracy_test(target_test, NN_output_cancer_test)


"""
Plot heatmap
"""

sns.set()
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis", cbar_kws={'label': 'Test Accuracy'},
            yticklabels=eta_vals, xticklabels=lmbd_vals)
ax.set_title("Test Accuracy logistic regression")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()

