import numpy as np
import json
from animal_guesser import ANIMALS

def apply_sigmoid(n):
    return 1 / (1 + np.exp(-n))

def load_model(filename="animal_model_weights.npz"):
    
    data = np.load(filename, allow_pickle=True)
    saved_weights = data["weights"]
    saved_biases = data["biases"]
    
    layers = []
    for layer_w, layer_b in zip(saved_weights, saved_biases):
        layer = []
        for w, b in zip(layer_w, layer_b):
            # Create a node using the shapes found in the file
            node = Node(weights=w, bias=b)
            layer.append(node)
        layers.append(layer)
        
    return NeuralNetwork(layers)


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
    def __init__(self, layers):
        self.layers = layers  # List of lists of Node objects
        self.layer_inputs = [] # Cache to store inputs for each layer during forward pass

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

if  __name__ == "__main__":

    filename = "practice.json"

    input_edges = edge_finder(filename)

    flattened_edges = input_edges.flatten()
    input_size = len(flattened_edges)
    hidden_size_1 = 32
    hidden_size_2 = 16

    #number of animals to classify
    animals = ANIMALS
    num_classes = len(animals)

    expected_targets_identifiers = {animal: np.zeros(num_classes) for i, animal in enumerate(animals)}
    for animal, target in expected_targets_identifiers.items():
        target[animals.index(animal)] = 1.0

    
    # Hidden Layer 1 (Takes 400 * 400 inputs per node from the flattened edge-detected image) unless the input size is changed
    layer1 = [
        Node(weights=np.random.randn(input_size) * 0.01, bias=0.0, identifier=f"H1_{i}")
        for i in range(hidden_size_1)
    ]

    # Hidden Layer 2 (Takes 32 inputs per node from Layer 1)
    layer2 = [
        Node(weights=np.random.randn(hidden_size_1) * 0.01, bias=0.0, identifier=f"H2_{i}")
        for i in range(hidden_size_2)
    ]

    # Output Layer (Takes 16 inputs per node from Layer 2)
    output_layer = [
        Node(weights=np.random.randn(hidden_size_2) * 0.01, bias=0.0, identifier=f"Out_{i}")
        for i in range(num_classes)
    ]

    # Combine into the Deep Neural Network
    deep_animal_net = NeuralNetwork([layer1, layer2, output_layer])

    # Run a forward prediction
    prediction = deep_animal_net.forward(flattened_edges)
    print("Initial Prediction Before Training:", prediction)


