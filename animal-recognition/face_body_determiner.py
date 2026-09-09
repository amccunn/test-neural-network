import numpy as np
import json

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
        
        # Backpropagation
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

if  __name__ == "__main__":

    with open("practice.json", "r") as f:
        pixelVals = np.array(list(json.load(f)))

    padded_pixelVals = np.pad(pixelVals, pad_width=1, mode='constant', constant_values=0)

    horizontal_edges = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1]
    ])

    vertical_edges = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ])


    lens_size = horizontal_edges.shape[0]
    pixel_size = pixelVals.shape[0]

    output = np.zeros((pixel_size, pixel_size))

    for y in range(pixel_size):

        for x in range(pixel_size):

            region = padded_pixelVals[y:y + lens_size, x:x + lens_size]

            horizontal_response = np.sum(region * horizontal_edges)
            vertical_response = np.sum(region * vertical_edges)

            combined_response = np.sqrt(horizontal_response**2 + vertical_response**2)

            output[y, x] = combined_response

    print(output)