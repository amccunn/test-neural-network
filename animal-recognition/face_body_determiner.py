import numpy as np
import json

with open("practice.txt", "r") as f:
    pixelVals = np.array(list(map(float, json.load(f))))

def apply_sigmoid(n):
    return 1 / (1 + np.exp(-n))

class Node:

    def __init__(self, weights, bias, activation_function=apply_sigmoid, identifier=None):
        self.weights = weights
        self.bias = bias
        self.activation_function = activation_function
        self.identifier = identifier

    def activate(self, inputs):
        # Calculate the weighted sum
        weighted_sum = np.dot(self.weights, inputs) + self.bias
        # Apply the activation function
        return self.activation_function(weighted_sum)

        

class NeuralNetwork:

    def __init__(self, layers):
        self.layers = layers  # List of lists of Node objects

    #forward pass of network
    def forward(self, inputs):

        for layer in self.layers:
            outputs = []

            for node in layer:

                output = node.activate(inputs)
                outputs.append(output)
            inputs = np.array(outputs)  # Outputs become inputs for the next layer
        return inputs

    def backward(self, inputs, expected_output, learning_rate):
        # Forward pass
        outputs = self.forward(inputs)
        
        # Calculate the error at the output layer
        error = expected_output - outputs
        
        # Backpropagation (simplified for demonstration)
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            new_errors = []
            for j, node in enumerate(layer):
                # Calculate the gradient
                gradient = error[j] * node.activation_function(outputs[j]) * (1 - node.activation_function(outputs[j]))
                # Update weights and bias
                node.weights += learning_rate * gradient * inputs
                node.bias += learning_rate * gradient
                new_errors.append(np.dot(node.weights, gradient))
            error = np.array(new_errors)  # Prepare error for the next layer

