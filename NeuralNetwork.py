
import numpy as np
import jax.numpy as jnp

from activation_functions import Activation_Functions
from cost_functions import *
from GD_class import*


class Dense_Layer:
    """
    Class for a dense layer in a neural network. Each layer consists of n_nodes with their own bias (float).
    Further each has weights vector of same length as n_inputs.
    """
    def __init__(self, n_inputs, n_nodes, activate_output, activation_function,weights=None, biases=None):

        self.n_inputs = n_inputs
        self.n_nodes = n_nodes
        self.activation_function = Activation_Functions(activation_function)
        self.constructed_from_scratch = True if weights or biases is None else False

        if self.constructed_from_scratch:
            # initiate weights of network randomly, scale by 0.1 to keep small
            # Initiate biases to 0.01
            self.weights = .1 * np.random.randn(self.n_inputs, self.n_nodes)
            self.biases = np.zeros((1, self.n_nodes)) + 0.01
        else:
            # Brukes om man allerede har trent en modell.
            # Legg til kontroll av dimensjoner og type
            self.weights = weights
            self.biases = biases

        self.activate_output = activate_output
        self.output = None
        self.output_pre_activation = None
        
        self.delta = None

    def forward_propagation(self, inputs):
        """
        Forward propagation of the network.
        :param inputs: Input data
        :return: Output of the layer
        """
        self.output_pre_activation = jnp.dot(inputs, self.weights) + self.biases

        if self.activate_output == True:
            self.output = self.activation_function.activation_function(self.output_pre_activation)
        else:
            self.output = self.output_pre_activation

        return self.output
    
    def backward_propagation(self, delta_next, weights_next, input_value, learning_rate, lmbd):
        
        da_dz = self.activation_function.grad_activation_function(self.output_pre_activation)
        
        self.delta = np.matmul(delta_next, weights_next.T) * da_dz
        
        dC_dW = np.matmul(input_value.T, self.delta)
        dC_db = np.sum(self.delta, axis=0)
        
        # Adding a regularization term
        dC_dW = dC_dW + lmbd * self.weights

        self.weights = self.weights - learning_rate * dC_dW
        self.biases = self.biases - learning_rate * dC_db
        
class Output_Layer(Dense_Layer):
    
    def __init__(self, cost_function, **kwargs):
        super().__init__(**kwargs)

        self.cost_function = cost_function
        
        self.delta = None


    def backward_propagation(self, input_value, learning_rate, lmbd):
        
        dC_da = self.cost_function.cost_function_grad(self.output)
        da_dz = self.activation_function.grad_activation_function(self.output_pre_activation)
        
        self.delta = dC_da * da_dz # Elementwise multiplication
        
        dC_dW = np.matmul(input_value.T, self.delta)
        dC_db = np.sum(self.delta, axis=0)
        
        # Adding a regularization term
        dC_dW = dC_dW + lmbd * self.weights
        
        self.weights = self.weights - learning_rate * dC_dW
        self.biases = self.biases - learning_rate * dC_db
    


class Neural_Network:

    def __init__(self, X, target, n_hidden_layers, n_hidden_nodes, n_outputs,
                 learning_rate=0.1,
                 lmbd=0.0,
                 activation_function='softmax', cost_function='CostCrossEntropy',
                 weights=None, biases=None,
                 classification_problem=False, initial_learning_rate=0.1, activate_output=True):

        # Initiate the basics of the network
        self.X = X
        self.target = target
        self.activation_function = activation_function
        self.cost_function = Cost_Functions(cost_function, self.target)

        self.activate_output = activate_output

        self.initial_learning_rate = learning_rate
        self.learning_rate=learning_rate
        self.lmbd = lmbd
        
        if weights or biases is None:
            self.construct_network_from_scratch = True
            self.construct_layer = self.construct_layer_from_scratch
            self.construct_output_layer = self.construct_output_layer_from_scratch
        else:
            self.construct_network_from_scratch = False
            self.construct_layer = self.construct_layer_from_previous_training
            self.construct_output_layer = self.construct_output_layer_from_previous_training
            self.weights = weights
            self.biases = biases

        # Initiate the hidden nodes and layers
        self.n_hidden_layers = n_hidden_layers
        if isinstance(n_hidden_nodes, int) or isinstance(n_hidden_nodes, float):
            n_hidden_nodes = [int(n_hidden_nodes)] * self.n_hidden_layers
        elif len(n_hidden_nodes) != self.n_hidden_layers:
            raise ValueError('n_hidden_nodes must be integer or a list of same length as number of hidden layers')

        self.n_hidden_nodes = n_hidden_nodes
        self.n_outputs = n_outputs

        self.hidden_layers = self.initiate_hidden_layers()
        self.output_layer = self.initiate_output_layer()
        
        self.classification_problem = classification_problem
        

    def __str__(self):
        return (f"Neural Network with {self.n_hidden_layers} hidden layers and {self.n_hidden_nodes} nodes per layer. "
                f"The activation function is {self.activation_function}.")

    def initiate_hidden_layers(self):
        """
        Initiates hidden layers of the network. If the nodes per layer is given as an integer, all hidden layers
        will have the same number of nodes, then we create a list of same length as n_hidden_layers.
        If the nodes per layer is given as a list it has to have same number of elements as n_hidden_layers.

        The first layer is initiated with n_inputs the same length as the input layer.
        The rest of the layers are initiated with n_inputs the same length as the previous hidden layer.

        :return: Hidden layers of the network
        """

        hidden_layers = []
        for i in range(self.n_hidden_layers):
            if i == 0:
                n_inputs = self.X.shape[1]
            else:
                n_inputs = self.n_hidden_nodes[i - 1]

            if self.construct_network_from_scratch:
                hidden_layers.append(self.construct_layer(n_inputs=n_inputs,
                                                          n_hidden_nodes=self.n_hidden_nodes[i],
                                                          activation_function=self.activation_function))
            else:
                hidden_layers.append(self.construct_layer(n_inputs=n_inputs,
                                                          n_hidden_nodes=self.n_hidden_nodes[i],
                                                          activation_function=self.activation_function,
                                                          layer_indx=i))

        return hidden_layers

    def initiate_output_layer(self):
        """
        Initiates the output layer of the network.
        :return: Output layer of the network
        """

        #for
        if len(self.hidden_layers) == 0:
            if self.construct_network_from_scratch:
                return self.construct_output_layer(n_inputs=len(self.X[0]),
                                                   n_hidden_nodes=self.n_outputs,
                                                   activation_function='sigmoid')
            else:
                return self.construct_output_layer(n_inputs=self.hidden_layers[-1].n_nodes,
                                                   n_hidden_nodes=self.n_outputs,
                                                   activation_function='sigmoid',
                                                   layer_indx=-1)
        else:
            if self.construct_network_from_scratch:
                return self.construct_output_layer(n_inputs=self.hidden_layers[-1].n_nodes,
                                                   n_hidden_nodes=self.n_outputs,
                                                   activation_function='sigmoid')
            else:
                return self.construct_output_layer(n_inputs=self.hidden_layers[-1].n_nodes,
                                                   n_hidden_nodes=self.n_outputs,
                                                   activation_function='sigmoid',
                                                   layer_indx=-1)
        
    def construct_output_layer_from_scratch(self, n_inputs, n_hidden_nodes, activation_function):
        """
        Constructs a layer without weights and biases.
        :param n_inputs: Number of inputs to the layer
        :param n_hidden_nodes: Number of nodes in the layer
        :param activation_function: Activation function of the layer
        :return: Layer with random weights and biases set to 0.01
        """
        return Output_Layer(cost_function=self.cost_function, 
                           n_inputs=n_inputs,
                           n_nodes=n_hidden_nodes,
                           activation_function=activation_function, activate_output=self.activate_output)

    def construct_output_layer_from_previous_training(self, n_inputs, n_hidden_nodes, activation_function, layer_indx):
        """
        Constructs a layer with weights and biases.
        :param n_inputs: Number of inputs to the layer
        :param n_hidden_nodes: Number of nodes in the layer
        :param activation_function: Activation function of the layer
        :param layer_indx: Index of the layer
        :return: Layer with weights and biases
        """
        return Output_Layer(cost_function=self.cost_function, 
                           n_inputs=n_inputs,
                           n_nodes=n_hidden_nodes,
                           activation_function=activation_function,
                           weights=self.weights[layer_indx],
                           biases=self.biases[layer_indx])

    def construct_layer_from_scratch(self, n_inputs, n_hidden_nodes, activation_function):
        """
        Constructs a layer without weights and biases.
        :param n_inputs: Number of inputs to the layer
        :param n_hidden_nodes: Number of nodes in the layer
        :param activation_function: Activation function of the layer
        :return: Layer with random weights and biases set to 0.01
        """
        return Dense_Layer(n_inputs=n_inputs,
                           n_nodes=n_hidden_nodes,
                           activation_function=activation_function, activate_output=self.activate_output)

    def construct_layer_from_previous_training(self, n_inputs, n_hidden_nodes, activation_function, layer_indx):
        """
        Constructs a layer with weights and biases.
        :param n_inputs: Number of inputs to the layer
        :param n_hidden_nodes: Number of nodes in the layer
        :param activation_function: Activation function of the layer
        :param layer_indx: Index of the layer
        :return: Layer with weights and biases
        """
        return Dense_Layer(n_inputs=n_inputs,
                           n_nodes=n_hidden_nodes,
                           activation_function=activation_function,
                           weights=self.weights[layer_indx],
                           biases=self.biases[layer_indx])
    
    def classify(self):
        self.output_layer.output = jnp.where(self.output_layer.output > 0.5, 1.0, 0.0)

    def feed_forward(self, X_test=None):
        for i in range(self.n_hidden_layers):
            if X_test is not None:
                if i == 0:
                    input_value = X_test
                else:
                    input_value = self.hidden_layers[i-1].output

                self.hidden_layers[i].forward_propagation(input_value)

        #self.output_layer.forward_propagation(self.hidden_layers[-1].output)

            if X_test is None:
                if i == 0:
                    input_value = self.X
                else:
                    input_value = self.hidden_layers[i-1].output

                self.hidden_layers[i].forward_propagation(input_value)
        #
        if len(self.hidden_layers) == 0:
            self.output_layer.forward_propagation(self.X)
        else:
            self.output_layer.forward_propagation(self.hidden_layers[-1].output)

        if self.classification_problem:
            self.classify()


        
    def feed_backward(self):

        if len(self.hidden_layers) == 0:
            self.output_layer.backward_propagation(input_value=self.X,
                                               learning_rate=self.learning_rate,
                                               lmbd=self.lmbd)
        else:
            self.output_layer.backward_propagation(input_value=self.hidden_layers[-1].output,
                                               learning_rate=self.learning_rate,
                                               lmbd=self.lmbd)
        
        for i in reversed(range(self.n_hidden_layers)):
            
            layer = self.hidden_layers[i]
            
            if i == (self.n_hidden_layers - 1):
                delta = self.output_layer.delta
                weights = self.output_layer.weights
                
            else:
                delta = self.hidden_layers[i+1].delta
                weights = self.hidden_layers[i+1].weights
                
            if i == 0:
                input_value = self.X
            else:
                input_value = self.hidden_layers[i-1].output
                
            
            layer.backward_propagation(delta_next=delta, 
                                       weights_next = weights,
                                       input_value = input_value,
                                       learning_rate=self.learning_rate,
                                       lmbd=self.lmbd)

    def predict(self, X_test):
        self.feed_forward(X_test=X_test)
        return self.output_layer.output

    def learning_schedule(self, method, iteration, num_iter):
        if method == "Fixed learning rate":
            pass
        elif method == "Linear decay":
            alpha = iteration / (num_iter)
            self.learning_rate = (1 - alpha) * self.initial_learning_rate \
                                 + alpha * self.initial_learning_rate * 0.01

    def train(self, num_iter=10, method='Fixed learning rate'):
        
        for i in range(num_iter):
            self.learning_schedule(method, iteration=i, num_iter=num_iter)
            self.feed_forward()
            self.feed_backward()

    def reset_weights(self):
        """
        Description:
        ------------
            Resets/Reinitializes the weights in order to train the network for a new problem.

        """

        self.construct_network_from_scratch = True
        
        
