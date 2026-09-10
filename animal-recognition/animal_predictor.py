import numpy as np
import json
from animal_guesser import ANIMALS

def apply_sigmoid(n):
    return 1 / (1 + np.exp(-n))


class Node:

    def __init__(self, weights, bias, activation_function=apply_sigmoid, identifier=None):
        self.weights = weights
        self.bias = bias
        self.activation_function = activation_function
        self.identifier = identifier

    def activate(self, inputs):
        weighted_sum = np.dot(self.weights, inputs) + self.bias
        return self.activation_function(weighted_sum)


class NeuralNetwork:

    def __init__(self, layers, model_name, learning_rate=0.01):

        self.layers = layers  # List of lists of Node objects
        self.layer_inputs = [] # Cache to store inputs for each layer during forward pass
        self.model_name = model_name
        self.learning_rate = learning_rate  # Default learning rate, can be adjusted as needed

    def forward(self, inputs):
        self.layer_inputs = [] # Reset cache
        current_input = inputs
        
        for layer in self.layers:
            self.layer_inputs.append(current_input) # Cache the input this layer sees
            outputs = []
            for node in layer:
                output = node.activate(current_input)
                outputs.append(output)
            current_input = np.array(outputs)
            
        return current_input

    def backward(self, inputs, expected_output, learning_rate):
        # 1. Run forward pass to populate caches
        outputs = self.forward(inputs)
        
        # 2. Calculate the error at the output layer
        error = expected_output - outputs
        
        # 3. Backpropagation loop going backwards through layers
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            layer_input = self.layer_inputs[i] # The exact input this layer received
            new_errors = np.zeros(len(layer_input))
            
            for j, node in enumerate(layer):
                node_output = node.activate(layer_input)
                
                # Gradient calculation using sigmoid derivative
                gradient = error[j] * node_output * (1 - node_output)
                
                # Accumulate error to pass back to the previous layer
                new_errors += node.weights * gradient
                
                # Update weights and bias
                node.weights += learning_rate * gradient * layer_input
                node.bias += learning_rate * gradient
                
            error = new_errors  # Pass error backward to the preceding layer

    def save_model(self):

        filename = self.model_name + ".npz"

        weights_list = []
        biases_list = []
        identifiers_list = []
        
        for layer in self.layers:

            # Extract weights and biases from each node in the layer
            layer_weights = [node.weights for node in layer]
            layer_biases = [node.bias for node in layer]  
            layer_identifiers = [node.identifier for node in layer]
            weights_list.append(np.array(layer_weights))
            biases_list.append(np.array(layer_biases))
            identifiers_list.append(np.array(layer_identifiers))
            
        # Save arrays into a single file
        np.savez(filename, weights=np.array(weights_list, dtype=object), biases=np.array(biases_list, dtype=object), identifiers=np.array(identifiers_list, dtype=object))

    def load_model(self, filename=None):

        if filename is None:

            filename = self.model_name + ".npz"

        data = np.load(filename, allow_pickle=True)
        saved_weights = data["weights"]
        saved_biases = data["biases"]
        saved_identifiers = data["identifiers"]
        
        layers = []
        for layer_w, layer_b, layer_identifiers in zip(saved_weights, saved_biases, saved_identifiers):
            layer = []
            for w, b, identifier in zip(layer_w, layer_b, layer_identifiers):
                # Create a node using the shapes found in the file
                node = Node(weights=w, bias=b, identifier=identifier)
                layer.append(node)
            layers.append(layer)
            
        self.layers = layers  # Update the model's layers with the loaded structure


def edge_finder(filename):

    with open(filename, "r") as f:
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

    edge_output = np.zeros((pixel_size, pixel_size))

    for y in range(pixel_size):

        for x in range(pixel_size):

            region = padded_pixelVals[y:y + lens_size, x:x + lens_size]

            horizontal_response = np.sum(region * horizontal_edges)
            vertical_response = np.sum(region * vertical_edges)

            combined_response = np.sqrt(horizontal_response**2 + vertical_response**2)

            edge_output[y, x] = combined_response

    return edge_output




